from stair.core.prompts import SYSTEM_PROMPT, USER_PROMPT


def test_system_prompt_is_a_non_empty_string():
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT.strip()) > 0


def test_user_prompt_has_expected_format_placeholders():
    rendered = USER_PROMPT.format(title="Test Book", toc="1 Introduction", question="What is this about?")
    assert "Test Book" in rendered
    assert "1 Introduction" in rendered
    assert "What is this about?" in rendered
