"""Database tests: tables, relationships, ordering and cascade delete."""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.models import Conversation, Message


def test_create_conversation_has_defaults(db_session):
    conv = Conversation(title="Hello")
    db_session.add(conv)
    db_session.commit()

    assert conv.id is not None
    assert conv.created_at is not None
    assert conv.updated_at is not None


def test_messages_are_linked_and_ordered(db_session):
    conv = Conversation(title="Chat")
    conv.messages.append(Message(role="user", content="Hi"))
    conv.messages.append(Message(role="assistant", content="Hello!"))
    db_session.add(conv)
    db_session.commit()

    loaded = db_session.get(Conversation, conv.id)
    assert [m.role for m in loaded.messages] == ["user", "assistant"]
    assert all(m.conversation_id == conv.id for m in loaded.messages)
    assert loaded.messages[0].conversation is loaded
    assert loaded.messages[0].created_at is not None


def test_deleting_conversation_removes_its_messages(db_session):
    conv = Conversation(title="Temp")
    conv.messages.append(Message(role="user", content="bye"))
    db_session.add(conv)
    db_session.commit()

    db_session.delete(conv)
    db_session.commit()

    assert db_session.scalars(select(Message)).all() == []


def test_message_requires_existing_conversation(db_session):
    db_session.add(Message(conversation_id=9999, role="user", content="orphan"))
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_messages_stay_separate_between_conversations(db_session):
    a, b = Conversation(title="A"), Conversation(title="B")
    a.messages.append(Message(role="user", content="in A"))
    b.messages.append(Message(role="user", content="in B"))
    db_session.add_all([a, b])
    db_session.commit()

    assert [m.content for m in a.messages] == ["in A"]
    assert [m.content for m in b.messages] == ["in B"]
