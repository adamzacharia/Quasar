from services.conversation_service import ConversationService


def test_messages_are_only_returned_to_conversation_owner(tmp_path):
    svc = ConversationService(str(tmp_path / "conversations.db"))
    alice_conv = svc.create_conversation("alice", "Alice chat")
    svc.save_message(alice_conv, "user", "Alice private message")

    alice_messages = svc.get_conversation_messages_for_user(alice_conv, "alice")
    bob_messages = svc.get_conversation_messages_for_user(alice_conv, "bob")

    assert alice_messages is not None
    assert alice_messages[0]["content"] == "Alice private message"
    assert bob_messages is None


def test_title_update_requires_conversation_owner(tmp_path):
    svc = ConversationService(str(tmp_path / "conversations.db"))
    alice_conv = svc.create_conversation("alice", "Original")

    assert svc.update_conversation_title_for_user(alice_conv, "bob", "Stolen") is False
    assert svc.update_conversation_title_for_user(alice_conv, "alice", "Updated") is True

    conversations = svc.get_user_conversations("alice")
    assert conversations[0]["title"] == "Updated"


def test_delete_requires_conversation_owner(tmp_path):
    svc = ConversationService(str(tmp_path / "conversations.db"))
    alice_conv = svc.create_conversation("alice", "Alice chat")
    svc.save_message(alice_conv, "user", "Keep this private")

    assert svc.delete_conversation_for_user(alice_conv, "bob") is False
    assert svc.get_conversation_messages_for_user(alice_conv, "alice") is not None

    assert svc.delete_conversation_for_user(alice_conv, "alice") is True
    assert svc.get_conversation_messages_for_user(alice_conv, "alice") is None
