"""Progress, appended to the file the server names in the environment.

The server sets `SADT_PROGRESS_FILE` to an absolute path and reads one JSON
object per line out of it, stamping the sequence number, the time and the state
itself. A supervised child inherits the variable and appends to the SAME file,
so a chain reports its progress with no plumbing at all.

Copied into this tool rather than shared -- see CONTRIBUTING.md on why there is
no package common to every tool. Stdlib only, and best effort: a tool that
cannot report its progress must still finish the run it was asked for, so every
failure here is swallowed.

**Never name a file in a message.** Position in the batch is progress; a file
name is patient metadata, and these messages are stored on the server and shown
to whoever is watching the run.
"""

import json
import os

VARIABLE = "SADT_PROGRESS_FILE"

# The server truncates a message to 200 characters; truncating here too is what
# keeps the write under PIPE_BUF, below which POSIX makes a write to an
# O_APPEND handle atomic -- which is what stops a supervised chain's events
# from interleaving into half a line each.
MAX_MESSAGE = 200
PIPE_BUF = 4096

# How many of its own items this tool is processing AT ONCE, right now. None
# until something says otherwise, and a record without it is IGNORED by the
# server rather than read as one -- which is what lets a tool declare a width
# for the phase that HAS one and stay silent through the serial phases around
# it. The server keeps the narrowest width it was told, and divides the run's
# peak by it to learn what one channel costs; telling it nothing means the
# whole peak is priced as a single channel, which over-reserves every run
# afterwards and then narrows them.
_width = None


def set_width(width):
    """Declare the width now in force, or None to stop declaring one.

    A MEASUREMENT, not a permission: `sup.channels()` says what a run may
    open, this says what it actually opened, and they are reported apart on
    purpose. A tool has phases of different widths -- AMASSS reads its cohort
    serially, predicts its structures side by side, then assembles serially --
    and declaring the permission through the serial phases would put a wide
    width in force where the peak actually happens, teaching a per-channel
    cost that is too low. That is the direction that ends in an
    out-of-memory.
    """
    global _width
    try:
        _width = None if width is None else max(1, int(width))
    except (TypeError, ValueError):
        _width = None


def emit(fraction, message):
    """Append one progress event. Never raises; does nothing when unset.

    `fraction` is 0..1, or None when the tool genuinely cannot know. None is
    the honest answer for an opaque phase -- an interpolated number reads as
    knowledge the tool does not have, and a bar that lies is worse than a bar
    that says nothing.
    """
    path = os.environ.get(VARIABLE)
    if not path:
        return  # nobody is watching: a checkout, a test, an older server
    try:
        if fraction is not None:
            fraction = round(min(1.0, max(0.0, float(fraction))), 4)
        record = {"fraction": fraction, "message": str(message)[:MAX_MESSAGE]}
        if _width is not None:
            record["width"] = _width
        line = json.dumps(record).encode("utf-8") + b"\n"
        if len(line) > PIPE_BUF:
            return  # a partial line would be unparsable; drop the event instead
        handle = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(handle, line)
        finally:
            os.close(handle)
    except Exception:  # noqa: BLE001 -- telemetry must never fail a run
        pass


def report(index, total, what, start=0.0, end=1.0):
    """Position in a batch: `report(14, 40, "scan")` is "scan 14 of 40", 32.5%.

    `index` is 1-based and names the item ABOUT to be processed, so the
    fraction is the share of the batch already behind it: it starts at 0 and
    never counts an item that is still running.

    `start` and `end` bound the slice of the whole run this loop occupies, for
    a tool whose batch is one phase among several -- without them a second
    phase sends the bar back to zero.

    `what` is a unit -- "scan", "patient", "mesh" -- never a file name.
    """
    done = (index - 1) / float(total) if total else 0.0
    emit(start + (end - start) * done, "{} {} of {}".format(what, index, total))
