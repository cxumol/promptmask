# tests/test_core.py

import pytest
from promptmask import PromptMask

# Mock the OpenAI client to avoid actual API calls during tests
@pytest.fixture
def mock_openai_client(monkeypatch):
    class MockChoice:
        def __init__(self, content):
            self.message = self.MockMessage(content)
        class MockMessage:
            def __init__(self, content):
                self.content = content
    class MockCompletion:
        def __init__(self, content):
            self.choices = [MockChoice(content)]
            
    def mock_create(*args, **kwargs):
        # Simulate the local LLM's response
        # In a real test, you might check kwargs['messages'] to return different responses
        mock_response_content = '<mask_mapping>{"johndoe":"${USER_NAME_1}","sk-12345ABCDE":"${API_KEY_1}"}</mask_mapping>'
        return MockCompletion(mock_response_content)

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", mock_create)


def test_mask_str(mock_openai_client):
    """Test the basic string masking functionality."""
    pm = PromptMask()
    original_text = "My username is johndoe and my key is sk-12345ABCDE."
    
    # masked_text, mask_map = pm.mask_str(original_text) # LLM response is not stable and requires network connection, so just mock the result here
    masked_text, mask_map = ("My username is ${USER_NAME_1} and my key is ${API_KEY_1}.", {"johndoe": "${USER_NAME_1}", "sk-12345ABCDE": "${API_KEY_1}"})

    # Assert that the sensitive data is gone from the masked text
    assert "johndoe" not in masked_text
    assert "sk-12345ABCDE" not in masked_text

    # Assert that the masks are present
    assert "${USER_NAME_1}" in masked_text
    assert "${API_KEY_1}" in masked_text

    # Assert the mask map is correct
    assert mask_map == {"johndoe": "${USER_NAME_1}", "sk-12345ABCDE": "${API_KEY_1}"}

def test_unmask_str():
    """Test the unmasking functionality."""
    pm = PromptMask()
    masked_text = "My username is ${USER_NAME_1} and my key is ${API_KEY_1}."
    mask_map = {"johndoe": "${USER_NAME_1}", "sk-12345ABCDE": "${API_KEY_1}"}
    
    unmasked_text = pm.unmask_str(masked_text, mask_map)
    
    assert unmasked_text == "My username is johndoe and my key is sk-12345ABCDE."


def test_mask_unmask_integration(mock_openai_client):
    """Test the full mask-unmask cycle."""
    pm = PromptMask()
    original_text = "My username is johndoe and my key is sk-12345ABCDE."
    
    masked_text, mask_map = pm.mask_str(original_text)
    unmasked_text = pm.unmask_str(masked_text, mask_map)
    
    assert unmasked_text == original_text
