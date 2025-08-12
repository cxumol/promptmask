import asyncio
from promptmask import PromptMask

async def main():
    masker = PromptMask()

    original_text = "Please process the visa application for Jensen Huang, passport number A12345678."

    # 1. Mask the text
    masked_text, mask_map = await masker.async_mask_str(original_text)

    print(f"Masked Text: {masked_text}")
    # Expected output (may vary): Masked Text: Please process the visa application for ${PERSON_NAME}, passport number ${PASSPORT_NUMBER}.
    
    print(f"Mask Map: {mask_map}")
    # Expected output: Mask Map: {"Jensen Huang": "'${PERSON_NAME}'", "A12345678": "'${PASSPORT_NUMBER}'"}

    # (Imagine sending masked_text to a remote API and getting a response)
    remote_response_text = "The visa application for ${PERSON_NAME} with passport ${PASSPORT_NUMBER} is now under review."

    # 2. Unmask the response
    unmasked_response = masker.unmask_str(remote_response_text, mask_map)
    print(f"Unmasked Response: {unmasked_response}")
    # Expected output: Unmasked Response: The visa application for Jensen Huang with passport A12345678 is now under review.

if __name__ == "__main__":
    asyncio.run(main())