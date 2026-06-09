import asyncio
import html
import os
import signal
import sys
from asyncio import Task
from asyncio.subprocess import Process
from typing import Any, ClassVar, Optional

from acp import NewSessionResponse, LoadSessionResponse
from acp.exceptions import RequestError
from acp.schema import AvailableCommand
from jupyter_ai_persona_manager import BasePersona
from jupyterlab_chat.models import Message

from .default_acp_client import JaiAcpClient
from .telemetry import emit_event, auto_emit_event


class BaseAcpPersona(BasePersona):
    _before_subprocess_future: ClassVar[Task[None] | None] = None
    """
    The task that blocks the agent subprocess from starting until resolved.

    By default this resolves immediately. Developers may define this task in
    `self.before_agent_subprocess()` - see method documentation for details.
    """

    _subprocess_future: ClassVar[Task[Process] | None] = None
    """
    The task that yields the agent subprocess once complete. This is a class
    attribute because multiple instances of the same ACP persona may share an
    ACP agent subprocess.

    Developers should always use `self.get_agent_subprocess()`.
    """

    _client_future: ClassVar[Task[JaiAcpClient] | None] = None
    """
    The future that yields the ACP Client once complete. This is a class
    attribute because multiple instances of the same ACP persona may share an
    ACP client as well. ACP agent subprocesses and clients map 1-to-1.

    Developers should always use `self.get_client()`.
    """

    _client_session_future: Task[NewSessionResponse | LoadSessionResponse]
    """
    The future that yields the ACP client session info. Each instance of an ACP
    persona has a unique session ID, i.e. each chat reserves a unique session.

    Developers should always call `self.get_session_response()` or `self.get_session_id()`.
    """

    _acp_slash_commands: list[AvailableCommand]
    """
    List of slash commands broadcast by the ACP agent in the current session.
    This attribute is set automatically by the default ACP client.
    """

    _MAX_HISTORY_MESSAGES: ClassVar[int] = 50
    """
    Maximum number of recent messages to include in the history context injected
    after load-session recovery. Caps prompt size to avoid exceeding agent
    context window limits.
    """

    def __init__(self, *args, executable: list[str], **kwargs):
        super().__init__(*args, **kwargs)

        self._executable = executable
        self._pending_session_recovery_context: bool = False
        self._was_initially_unauthenticated: bool = False

        # Ensure each subclass has its own subprocess and client by checking if the
        # class variable is defined directly on this class (not inherited)
        if (
            "_before_subprocess_future" not in self.__class__.__dict__
            or self.__class__._before_subprocess_future is None
        ):
            self.__class__._before_subprocess_future = self.event_loop.create_task(
                self.before_agent_subprocess()
            )
        if (
            "_subprocess_future" not in self.__class__.__dict__
            or self.__class__._subprocess_future is None
        ):
            self.__class__._subprocess_future = self.event_loop.create_task(
                self._init_agent_subprocess()
            )
        if (
            "_client_future" not in self.__class__.__dict__
            or self.__class__._client_future is None
        ):
            self.__class__._client_future = self.event_loop.create_task(
                self._init_client()
            )

        self._client_session_future = self.event_loop.create_task(
            self._init_client_session()
        )
        self._acp_slash_commands = []

    async def before_agent_subprocess(self) -> None:
        """
        Defines a task that blocks the ACP agent subprocess from starting until
        resolved. This is useful for when the ACP agent subprocess cannot be
        started until certain requirements are met (e.g. Kiro).

        The `BaseAcpPersona` does not implement this method by default.
        Subclasses are expected to provide a custom implementation of this
        method if required.
        """
        return None

    async def _init_agent_subprocess(
        self, env: Optional[dict[str, str]] = None
    ) -> Process:
        # Wait until user is authenticated
        await self._before_subprocess_future
        self.log.info("Spawning ACP agent subprocess for '%s'.", self.__class__.__name__)

        if sys.platform != "win32":
            # Unix/macOS: use asyncio.create_subprocess_exec() directly.
            # ProactorEventLoop (or any loop supporting subprocess pipes) is available.
            kwargs: dict[str, Any] = dict(
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=sys.stderr,
                limit=50 * 1024 * 1024,
                start_new_session=True,
            )
            if env is not None:
                kwargs["env"] = env
            process = await asyncio.create_subprocess_exec(*self._executable, **kwargs)

        else:
            # Windows workaround:
            #
            # jupyter_server intentionally switches the event loop policy to
            # WindowsSelectorEventLoopPolicy for Tornado compatibility
            # (see jupyter_server/utils.py: maybe_patch_ioloop()).
            # SelectorEventLoop does NOT support asyncio.create_subprocess_exec()
            # with PIPE on Windows -- it raises NotImplementedError because Windows
            # cannot select() on pipes (only on sockets).
            #
            # ProactorEventLoop (IOCP-based) supports pipe subprocesses, but is
            # incompatible with Tornado 6.x which Jupyter depends on.
            #
            # Workaround: use subprocess.Popen (blocking) and bridge stdio via:
            #   - A daemon thread that calls readline() on stdout (blocking I/O)
            #     and feeds data into an asyncio.Queue via call_soon_threadsafe().
            #   - FakeStreamReader/FakeStreamWriter subclasses that pass
            #     isinstance(x, asyncio.StreamReader/StreamWriter) checks required
            #     by acp.client.connection.ClientSideConnection, while internally
            #     using the Queue-based bridge instead of the real transport.
            #
            # NOTE: read(n) does NOT work here -- Windows pipe buffering causes it
            # to block until the buffer is full. readline() is required because
            # ACP uses newline-delimited JSON (JSON-Lines), so each response is
            # exactly one line.
            #
            # TODO: If goose/other ACP agents ever support a --port TCP listen
            # mode, replace this with asyncio.open_connection() which works
            # natively on SelectorEventLoop.

            import subprocess
            import threading

            popen_kwargs: dict[str, Any] = dict(
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                start_new_session=True,
            )
            if env is not None:
                popen_kwargs["env"] = env

            popen = subprocess.Popen(list(self._executable), **popen_kwargs)
            loop = asyncio.get_event_loop()

            class _FakeStreamReader(asyncio.StreamReader):
                """
                Subclass of asyncio.StreamReader that passes isinstance() checks
                but uses an asyncio.Queue fed by a background thread instead of
                the real asyncio transport (which requires ProactorEventLoop on
                Windows).
                """
                def __init__(self, limit: int = 2 ** 16):
                    # Do NOT call super().__init__() -- it would attempt to bind to
                    # the current event loop transport, which fails on
                    # SelectorEventLoop with pipe handles.
                    self._queue: asyncio.Queue = None  # lazy-initialized
                    self._buf = b""
                    self._eof = False

                def _get_queue(self) -> asyncio.Queue:
                    # Lazy initialization ensures the Queue is created on the
                    # running event loop, not at construction time.
                    if self._queue is None:
                        self._queue = asyncio.Queue()
                    return self._queue

                def feed_data_threadsafe(
                    self, loop: asyncio.AbstractEventLoop, data: bytes
                ) -> None:
                    loop.call_soon_threadsafe(self._get_queue().put_nowait, data)

                def feed_eof_threadsafe(
                    self, loop: asyncio.AbstractEventLoop
                ) -> None:
                    loop.call_soon_threadsafe(self._get_queue().put_nowait, None)

                async def readline(self) -> bytes:
                    while b"\n" not in self._buf and not self._eof:
                        item = await self._get_queue().get()
                        if item is None:
                            self._eof = True
                            break
                        self._buf += item
                    if b"\n" in self._buf:
                        idx = self._buf.index(b"\n")
                        line, self._buf = self._buf[: idx + 1], self._buf[idx + 1 :]
                    else:
                        line, self._buf = self._buf, b""
                    return line

                async def readuntil(self, separator: bytes = b"\n") -> bytes:
                    while separator not in self._buf and not self._eof:
                        item = await self._get_queue().get()
                        if item is None:
                            self._eof = True
                            break
                        self._buf += item
                    if separator in self._buf:
                        idx = self._buf.index(separator) + len(separator)
                        line, self._buf = self._buf[:idx], self._buf[idx:]
                    else:
                        line, self._buf = self._buf, b""
                    return line

                async def readexactly(self, n: int) -> bytes:
                    while len(self._buf) < n and not self._eof:
                        item = await self._get_queue().get()
                        if item is None:
                            raise asyncio.IncompleteReadError(self._buf, n)
                        self._buf += item
                    data, self._buf = self._buf[:n], self._buf[n:]
                    return data

                async def read(self, n: int = -1) -> bytes:
                    if not self._buf and not self._eof:
                        item = await self._get_queue().get()
                        if item is None:
                            self._eof = True
                        else:
                            self._buf += item
                    if n == -1:
                        data, self._buf = self._buf, b""
                    else:
                        data, self._buf = self._buf[:n], self._buf[n:]
                    return data

                def at_eof(self) -> bool:
                    return self._eof and not self._buf

            class _FakeStreamWriter(asyncio.StreamWriter):
                """
                Subclass of asyncio.StreamWriter that passes isinstance() checks
                but writes synchronously to the Popen stdin pipe instead of using
                the real asyncio transport.

                write() flushes immediately because the ACP sender calls drain()
                after write(), and a buffered-but-unflushed write would cause
                Goose to wait indefinitely for a complete JSON-Lines message.
                """

                class _DummyTransport:
                    """
                    Minimal transport stub to satisfy asyncio.StreamWriter.__del__,
                    which references self._transport without calling any methods on it.
                    Any method that may be called during cleanup is implemented as a no-op.
                    """
                    def is_closing(self) -> bool:
                        return True

                    def close(self) -> None:
                        pass

                    def get_extra_info(self, name: str, default=None):
                        return default

                def __init__(self, popen_stdin, loop: asyncio.AbstractEventLoop):
                    # Do NOT call super().__init__() -- requires a real transport.
                    self._popen_stdin = popen_stdin
                    self._loop = loop
                    # Satisfy asyncio.StreamWriter.__del__ which accesses self._transport
                    self._transport = self._DummyTransport()

                def write(self, data: bytes) -> None:
                    self._popen_stdin.write(data)
                    self._popen_stdin.flush()

                def writelines(self, data) -> None:
                    for chunk in data:
                        self.write(chunk)

                async def drain(self) -> None:
                    pass  # Already flushed synchronously in write()

                def close(self) -> None:
                    try:
                        self._popen_stdin.close()
                    except Exception:
                        pass

                async def wait_closed(self) -> None:
                    pass

                def is_closing(self) -> bool:
                    return False

                def get_extra_info(self, name: str, default=None):
                    return default

            stdout_reader = _FakeStreamReader(limit=50 * 1024 * 1024)
            stdin_writer = _FakeStreamWriter(popen.stdin, loop)

            stop_event = threading.Event()

            def _do_terminate():
                """Ensure Goose subprocess and bridge thread are cleaned up."""
                stop_event.set()
                try:
                    popen.stdout.close()
                except Exception:
                    pass
                try:
                    if popen.returncode is None:
                        popen.kill()
                except Exception:
                    pass

            import atexit
            atexit.register(_do_terminate)

            def _stdout_bridge() -> None:
                # Runs in a daemon thread. Calls readline() (blocking) on the
                # Popen stdout pipe and forwards each line to the async reader.
                #
                # IMPORTANT: read(n) must NOT be used here. On Windows, pipe reads
                # block until n bytes are available in the buffer regardless of
                # whether a complete message has arrived. Since ACP messages are
                # newline-terminated, readline() is both correct and necessary.
                try:
                    while not stop_event.is_set():
                        line = popen.stdout.readline()
                        if not line:
                            stdout_reader.feed_eof_threadsafe(loop)
                            break
                        stdout_reader.feed_data_threadsafe(loop, line)
                except Exception:
                    stdout_reader.feed_eof_threadsafe(loop)

            threading.Thread(target=_stdout_bridge, daemon=True).start()

            class _PortableProcess:
                """
                Duck-typed replacement for asyncio.subprocess.Process.
                Wraps a subprocess.Popen instance with FakeStreamReader/Writer
                so that the rest of the ACP client stack is unaware of the
                Windows workaround.
                """
                def __init__(self):
                    # Note: ClientSideConnection(client, input_stream, output_stream)
                    # expects input_stream=StreamWriter (stdin) and
                    # output_stream=StreamReader (stdout) from the agent's perspective.
                    self.stdin = stdin_writer
                    self.stdout = stdout_reader
                    self.pid = popen.pid
                    self._popen = popen

                @property
                def returncode(self):
                    return popen.returncode

                async def wait(self):
                    return await loop.run_in_executor(None, popen.wait)

                def terminate(self):
                    _do_terminate()

                def kill(self):
                    _do_terminate()

            process = _PortableProcess()

        self.log.info("Spawned ACP agent subprocess for '%s'.", self.__class__.__name__)
        return process

    @auto_emit_event("acp_server_init")
    async def _init_client(self) -> JaiAcpClient:
        agent_subprocess = await self.get_agent_subprocess()
        client = JaiAcpClient(
            agent_subprocess=agent_subprocess, event_loop=self.event_loop
        )
        self.log.info("Initialized ACP client for '%s'.", self.__class__.__name__)
        return client

    def _get_existing_sessions(self) -> dict[str, str]:
        """
        Returns ACP session IDs from this chat's metadata, keyed by persona ID.
        """
        sessions = self.ychat.get_metadata().get("acp_session_ids", {})
        return sessions

    def _record_new_session(self, new_session_id: str) -> None:
        """
        Adds a new ACP session ID into this chat's metadata. Always use this
        method to avoid deleting other clients' sessions.

        Updates the `ychat._ydoc["metadata"]` shared type internally.
        """
        existing_session_ids = self._get_existing_sessions()
        self.ychat.set_metadata(
            "acp_session_ids", {**existing_session_ids, self.id: new_session_id}
        )

    @auto_emit_event("acp_session_init", lambda self: {"session_operation": "load"})
    async def _load_session(self, client, existing_session_id) -> LoadSessionResponse:
        response = await client.load_session(
            persona=self, session_id=existing_session_id
        )
        self.log.info(
            "Loaded existing ACP client session for '%s' with ID '%s'.",
            self.__class__.__name__,
            existing_session_id,
        )
        return response

    @auto_emit_event("acp_session_init", lambda self: {"session_operation": "new"})
    async def _create_session(self, client) -> NewSessionResponse:
        response = await client.create_session(persona=self)
        self.log.info(
            "Initialized new ACP client session for '%s' with ID '%s'.",
            self.__class__.__name__,
            response.session_id,
        )
        self._record_new_session(response.session_id)
        return response

    async def _init_client_session(self) -> NewSessionResponse | LoadSessionResponse:
        # get client
        client = await self.get_client()

        # check for an existing session ID
        existing_session_id = self._get_existing_sessions().get(self.id, None)
        supports_session_load = (await client.get_agent_capabilities()).load_session

        if existing_session_id and supports_session_load:
            try:
                return await self._load_session(client, existing_session_id)
            except Exception:
                self.log.warning(
                    "Failed to load ACP client session for '%s' with ID '%s'; "
                    "creating a new session.",
                    self.__class__.__name__,
                    existing_session_id,
                    exc_info=True,
                )
                self._pending_session_recovery_context = True
                return await self._create_session(client)
        else:
            response = await self._create_session(client)

            # If the user was initially unauthenticated and the session was
            # blocked on auth (e.g. Kiro, Gemini), proactively resume their
            # original request now that the session is ready.
            if self._was_initially_unauthenticated:
                self._was_initially_unauthenticated = False
                await self._resume_after_auth(client, response.session_id)

            return response

    async def _resume_after_auth(
        self, client: JaiAcpClient, session_id: str
    ) -> None:
        """
        After the user signs in, send a hidden prompt with chat history and a
        prescribed message template for the agent to follow.
        """
        history = self._build_history_context(
            preamble=(
                "You just became available after the user signed in. "
                "Here are the messages they sent while you were unavailable:"
            )
        )
        if history:
            prompt = (
                history + "\n\n"
                "Display the following message to the user, filling in the "
                "bracketed section with a brief summary of their request:\n\n"
                "\"I'm logged in now. I see you asked about [brief summary of "
                "their request]. Would you like me to help you with this now?\""
            )
        else:
            prompt = (
                "Display the following message to the user exactly as written:\n\n"
                "\"I'm logged in now and ready to help. What can I do for you?\""
            )

        await client.prompt_and_reply(
            session_id=session_id,
            prompt=prompt,
            root_dir=self.parent.root_dir,
        )

    async def get_agent_subprocess(self) -> asyncio.subprocess.Process:
        """
        Safely returns the ACP agent subprocess for this persona.
        """
        return await self.__class__._subprocess_future

    async def get_client(self) -> JaiAcpClient:
        """
        Safely returns the ACP client for this persona.
        """
        return await self.__class__._client_future

    async def get_session_response(self) -> NewSessionResponse | LoadSessionResponse:
        """
        Safely returns the ACP session response for this chat.
        """
        return await self._client_session_future

    async def get_session_id(self) -> str:
        """
        Safely returns the ACP session ID assigned to this chat.
        """
        await self._client_session_future
        # session ID should always be stored in chat metadata after client
        # session was created or loaded.
        session_ids = self._get_existing_sessions()
        assert self.id in session_ids
        return session_ids[self.id]

    async def is_authed(self) -> bool:
        """
        Returns whether the client is authenticated to use this agent. Returns
        `True` by default. Subclasses should override this if possible.
        """
        return True

    async def handle_no_auth(self, message: Message) -> None:
        """
        Method called when the persona receives a message while the user is not
        authenticated. Sets the `_was_initially_unauthenticated` flag so the
        agent can proactively resume the user's request after signing in.

        Subclasses should call `await super().handle_no_auth(message)` first,
        then send a custom message asking the user to log in and perform any
        additional setup (e.g. opening a login terminal).
        """
        self._was_initially_unauthenticated = True
        self.log.warning(
            "[%s] Received message while unauthenticated.", self.__class__.__name__
        )

    def _build_history_context(
        self,
        exclude_id: str | None = None,
        preamble: str = (
            "The previous ACP session could not be loaded. Use this recent chat "
            "transcript as historical context for continuity."
        ),
    ) -> str:
        """
        Builds a plain-text summary of recent chat history for context injection.
        Caps at _MAX_HISTORY_MESSAGES to avoid exceeding agent context window
        limits.
        """
        all_messages = self.ychat.get_messages()
        recent = [
            m for m in all_messages
            if not m.deleted and m.id != exclude_id
        ][-self._MAX_HISTORY_MESSAGES:]
        if not recent:
            return ""
        users = self.ychat.get_users()
        lines = []
        for msg in recent:
            user = users.get(msg.sender)
            name = user.display_name if user else msg.sender
            lines.append(f"{name}: {msg.body}")
        return (
            preamble + "\n"
            "<conversation_history>\n"
            + "\n".join(lines)
            + "\n</conversation_history>"
        )

    @auto_emit_event("acp_chat_message")
    async def process_message(self, message: Message) -> None:
        """
        A default implementation for the `BasePersona.process_message()` method
        for ACP agents.

        This method may be overriden by child classes.
        """
        # If not authenticated, return early
        if not await self.is_authed():
            await self.handle_no_auth(message)
            return

        # If the user was previously unauthenticated, proactively resume their
        # original request instead of processing this message normally.
        if self._was_initially_unauthenticated:
            self._was_initially_unauthenticated = False
            client = await self.get_client()
            session_id = await self.get_session_id()
            await self._resume_after_auth(client, session_id)
            return

        client = await self.get_client()
        session_id = await self.get_session_id()
        prompt = message.body.strip()

        if self._pending_session_recovery_context:
            self._pending_session_recovery_context = False
            history = self._build_history_context(exclude_id=message.id)
            if history:
                emit_event(
                    self.event_logger,
                    "acp_session_recovery",
                    "success",
                    {"persona_class": self.__class__.__name__},
                )
                prompt = history + "\n\nCurrent user message:\n" + prompt

        # Resolve attachments from YChat by ID
        attachments: list[dict] | None = None
        if message.attachments:
            all_attachments = self.ychat.get_attachments()
            resolved = []
            for aid in message.attachments:
                raw = all_attachments.get(aid)
                if raw is None:
                    self.log.warning("Attachment %s not found in YChat", aid)
                    continue
                resolved.append(raw)
            attachments = resolved or None

        await client.prompt_and_reply(
            session_id=session_id,
            prompt=prompt,
            attachments=attachments,
            root_dir=self.parent.root_dir,
        )

    @property
    def acp_slash_commands(self) -> list[AvailableCommand]:
        """
        Returns the list of slash commands advertised by the ACP agent in the
        current session.

        This initializes to an empty list, and should be updated **only** by the
        ACP client upon receiving a `session/update` request containing an
        `AvailableCommandsUpdate` payload from the ACP agent.
        """
        return self._acp_slash_commands

    @acp_slash_commands.setter
    def acp_slash_commands(self, commands: list[AvailableCommand]):
        self.log.info(
            "Setting %d slash commands for '%s' in room '%s'.",
            len(commands),
            self.name,
            self.parent.room_id,
        )
        self._acp_slash_commands = commands

    async def handle_uncaught_exception(self, exc: Exception) -> None:
        """Show structured error info for ACP RequestError inside the standard dropdown."""
        if not isinstance(exc, RequestError):
            await super().handle_uncaught_exception(exc)
            return

        import json
        import traceback

        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        error_msg = str(exc)
        if len(error_msg) > 120:
            error_msg = error_msg[:120] + "…"
        summary = f"Error {exc.code}: {html.escape(error_msg)}"

        # Build inner content
        sections = [f"**Error code:** {exc.code}\n\n**Message:** {html.escape(str(exc))}"]

        if exc.data is not None:
            try:
                data_str = json.dumps(exc.data, indent=2)
            except (TypeError, ValueError):
                data_str = str(exc.data)
            sections.append(f"**Data:**\n\n```json\n{data_str}\n```")

        sections.append(f"**Traceback:**\n\n```\n{tb}```")

        inner = "\n\n".join(sections)

        body = (
            f"An error occurred while processing your message.\n\n"
            f'<details class="jp-jai-error-details">\n'
            f"<summary>Error details ({summary})</summary>\n\n"
            f"{inner}\n"
            f"</details>"
        )
        self.send_message(body)

    async def shutdown(self):
        if getattr(self, "_shutting_down", False):
            return
        self._shutting_down = True
        await super().shutdown()
        await self._shutdown()

    async def _shutdown(self):
        self.log.info("[shutdown] Starting for '%s'.", self.__class__.__name__)

        # Cancel any pending startup futures to avoid hanging on auth-gated
        # personas (e.g. Kiro, Gemini) that never finished startup.
        for future in [
            self.__class__._before_subprocess_future,
            self.__class__._subprocess_future,
            self.__class__._client_future,
            self._client_session_future,
        ]:
            if isinstance(future, Task) and not future.done():
                future.cancel()

        # Step 1: Session cleanup
        try:
            client = await self.get_client()
            session_id = await self.get_session_id()
            await client.end_session(session_id)
            self.log.info(
                "[shutdown] Step 1: session ended for '%s'.",
                self.__class__.__name__,
            )
        except asyncio.CancelledError:
            pass
        except Exception:
            self.log.warning(
                "[shutdown] Step 1: failed for '%s'.",
                self.__class__.__name__,
                exc_info=True,
            )

        # Skip connection/subprocess teardown if other sessions are still active
        try:
            client = await self.get_client()
            if client.list_sessions():
                self.log.info(
                    "[shutdown] Other sessions still active, skipping subprocess teardown for '%s'.",
                    self.__class__.__name__,
                )
                return
        except (asyncio.CancelledError, Exception):
            pass

        # Step 2: Close connection
        try:
            client = await self.get_client()
            conn = await client.get_connection()
            await conn.close()
            self.log.info(
                "[shutdown] Step 2: connection closed for '%s'.",
                self.__class__.__name__,
            )
        except asyncio.CancelledError:
            pass
        except Exception:
            self.log.warning(
                "[shutdown] Step 2: failed for '%s'.",
                self.__class__.__name__,
                exc_info=True,
            )


        # Step 3: Stop the subprocess and its process tree.
        # Use psutil for cross-platform process tree termination instead of
        # os.getpgid()/os.killpg() which are Unix-only and fail on Windows.
        # psutil is already a transitive dependency via jupyter_client and
        # ipykernel, so this introduces no new packages in practice.
        try:
            import contextlib
            import psutil
            subprocess = await self.get_agent_subprocess()
            try:
                parent = psutil.Process(subprocess.pid)
                children = parent.children(recursive=True)
                for child in children:
                    with contextlib.suppress(psutil.NoSuchProcess):
                        child.terminate()
                parent.terminate()
                try:
                    await asyncio.wait_for(subprocess.wait(), timeout=5.0)
                    self.log.info(
                        "[shutdown] Step 3: subprocess terminated for '%s'.",
                        self.__class__.__name__,
                    )
                except asyncio.TimeoutError:
                    for child in children:
                        with contextlib.suppress(psutil.NoSuchProcess):
                            child.kill()
                    with contextlib.suppress(psutil.NoSuchProcess):
                        parent.kill()
                    self.log.info(
                        "[shutdown] Step 3: subprocess killed after timeout for '%s'.",
                        self.__class__.__name__,
                    )
            except psutil.NoSuchProcess:
                self.log.info(
                    "[shutdown] Step 3: subprocess already dead for '%s'.",
                    self.__class__.__name__,
                )
        except asyncio.CancelledError:
            pass
        except Exception:
            self.log.warning(
                "[shutdown] Step 3: failed for '%s'.",
                self.__class__.__name__,
                exc_info=True,
            )

        # Reset class attributes to `None` after cleaning up the global
        # resources they store
        self.__class__._before_subprocess_future = None
        self.__class__._subprocess_future = None
        self.__class__._client_future = None

        self.log.info("[shutdown] Complete for '%s'.", self.__class__.__name__)

    @property
    def event_logger(self):
        """Return the Jupyter EventLogger, or None if unavailable."""
        try:
            from jupyter_events import EventLogger

            extension_app = self.parent.parent  # ExtensionApp instance
            event_logger: EventLogger = extension_app.serverapp.event_logger
            return event_logger
        except Exception:
            self.log.warning("EventLogger unavailable; event logging will be skipped.")
            return None
