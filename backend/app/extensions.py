"""Extension points for later phases.

Phase 1 calls these hooks and keeps them side-effect free.
Phase 3.2 grounded answers live on /api/knowledge/ask (not chat memory).
Agents, MCP, and authentication can plug in here later.
"""


def augment_instructions(instructions: str, *, conversation_id: str) -> str:
    """Return the system instructions for a turn.

    Conversational RAG memory can append retrieved context here in a later phase.
    """

    del conversation_id
    return instructions


def resolve_operator(authorization_header: str | None = None) -> str:
    """Identify the caller.

    Phase 1 is a single local operator. A later auth phase can validate
    the authorization header and return a real principal.
    """

    del authorization_header
    return "local-operator"
