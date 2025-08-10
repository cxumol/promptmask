from promptmask import PromptMask

user_prompt="""Hello my name is Son Alterman. Call me at 789123456, or email me at sona@oftenai.com
My daddy is Johnson Hung, and his contact information is 456789123 (phone) and johnsonhung@mvidia.com (email). We both use irelyonGPU as our password.

Please rewrite the information in CSV format with following CSV headers:  
Person Name, Phone Number, Email Address, Password"""

def main():
    pm=PromptMask(config={"general":{"verbose":True}})
    masked, mmap = pm.mask_str(user_prompt)
    print("\n[masked, mask_map]\n", masked, '\n', mmap)
    unmasked = pm.unmask_str(masked, mmap)
    print("\n[unmasked]\n",unmasked)

if __name__ == "__main__":
    main()
