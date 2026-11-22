"""
Unit tests for MCP Action Scope granular policy filtering.
"""

from mcp_registry.factory import MCPToolsetFactory


class DummyServer:
    def __init__(self, action_scope, cached_tools=None, allowed_tools=None, blocked_tools=None):
        self.action_scope = action_scope
        self.cached_tool_schemas = cached_tools or []
        self.allowed_tools = allowed_tools
        self.blocked_tools = blocked_tools or []


SAMPLE_TOOLS = [
    {"name": "jira_search_issues", "description": "Search issues"},
    {"name": "jira_get_issue", "description": "Get issue details"},
    {"name": "jira_create_issue", "description": "Create a new issue"},
    {"name": "jira_update_issue", "description": "Update an existing issue"},
    {"name": "jira_delete_issue", "description": "Delete an issue"},
]


def test_read_only_scope():
    factory = MCPToolsetFactory()
    server = DummyServer(action_scope="read_only", cached_tools=SAMPLE_TOOLS)
    filtered = factory._apply_action_policy(server, allowed=None, blocked=set())

    assert "jira_search_issues" in filtered
    assert "jira_get_issue" in filtered
    assert "jira_create_issue" not in filtered
    assert "jira_update_issue" not in filtered
    assert "jira_delete_issue" not in filtered


def test_read_and_create_scope():
    """Verify that 'read,create' permits search + creation, but blocks updates and deletes."""
    factory = MCPToolsetFactory()
    server = DummyServer(action_scope="read,create", cached_tools=SAMPLE_TOOLS)
    filtered = factory._apply_action_policy(server, allowed=None, blocked=set())

    assert "jira_search_issues" in filtered
    assert "jira_get_issue" in filtered
    assert "jira_create_issue" in filtered
    assert "jira_update_issue" not in filtered
    assert "jira_delete_issue" not in filtered


def test_compound_scope_with_plus_sign():
    """Verify that 'read+create+update' permits everything except delete."""
    factory = MCPToolsetFactory()
    server = DummyServer(action_scope="read+create+update", cached_tools=SAMPLE_TOOLS)
    filtered = factory._apply_action_policy(server, allowed=None, blocked=set())

    assert "jira_search_issues" in filtered
    assert "jira_create_issue" in filtered
    assert "jira_update_issue" in filtered
    assert "jira_delete_issue" not in filtered


def test_blocked_tools_always_overrides():
    """Verify that explicitly blocked tools are never exposed regardless of scope."""
    factory = MCPToolsetFactory()
    server = DummyServer(
        action_scope="read,create",
        cached_tools=SAMPLE_TOOLS,
        blocked_tools=["jira_create_issue"]
    )
    filtered = factory._apply_action_policy(server, allowed=None, blocked={"jira_create_issue"})

    assert "jira_search_issues" in filtered
    assert "jira_create_issue" not in filtered
