import os
import sys
import getpass


def setup_openai_api_key():
    """Set up the OpenAI API key for use with the demo."""
    # Check for OpenAI API key (in environment variables first)
    api_key = os.environ.get("OPENAI_API_KEY")

    # If not in environment variables, try to read from .env file
    if not api_key:
        env_path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path):
            try:
                # Read .env file
                with open(env_path) as env_file:
                    lines = env_file.readlines()
                    for line in lines:
                        if line.strip().startswith("OPENAI_API_KEY="):
                            # Extract key from line (handle quotes)
                            key_part = line.strip().split("=", 1)[1]
                            api_key = key_part.strip("\"'")
                            if api_key:
                                os.environ["OPENAI_API_KEY"] = api_key
                                print("✅ Found OpenAI API key in .env file")
                                break
            except Exception as e:
                print(f"⚠️  Error reading .env file: {e}")

    # If still no key, prompt the user
    if not api_key:
        print("\n🔐 API Key Security Information:")
        print("--------------------------------")
        print(
            "• Your OpenAI API key is stored locally only in the .env file or temporary environment variable"
        )
        print("• The key is never transmitted to any third-party servers")
        print(
            "• API calls are only made directly to OpenAI's servers using your key via the official OpenAI Python Package\n"
        )
        print(
            "• The source code is open and can be reviewed under: https://github.com/TheLucasSchwarz/zeroshotENGINE\n"
        )
        print("• For security, never commit the .env file to public repositories")
        print("• When using .env and git, add the .env to you gitignore")
        print("• You can delete your API key from .env at any time")
        print("--------------------------------")

        # Ask how user wants to provide the key
        print("\n❌ No OpenAI API key found in environment or .env file.")
        print(
            "⚠️  Note: Using the OpenAI API will incur charges to your OpenAI account\n"
        )

        print("Please provide your OpenAI API key to continue.")

        print("\n❔ How would you like to provide your API key?")
        print("     1. Enter for this session only (not saved)")
        print("     2. Save to .env file for future use")
        print("     3. Exit and add manually later")

        key_choice = ""
        while key_choice not in ["1", "2", "3"]:
            key_choice = input("\nEnter your choice (1, 2, or 3): ").strip()

        if key_choice == "1":
            # Use key for this session only
            api_key = getpass.getpass(
                "Enter your OpenAI API key (input hidden): "
            ).strip()
            os.environ["OPENAI_API_KEY"] = api_key
            print("✅ API key set for this session")

        elif key_choice == "2":
            # Save to .env file
            api_key = getpass.getpass(
                "Enter your OpenAI API key (input hidden): "
            ).strip()
            os.environ["OPENAI_API_KEY"] = api_key

            env_path = os.path.join(os.getcwd(), ".env")
            try:
                # Check if file exists and if so, append or replace
                if os.path.exists(env_path):
                    # Read existing content
                    with open(env_path) as env_file:
                        lines = env_file.readlines()

                    # Check if OPENAI_API_KEY already exists
                    key_exists = False
                    for i, line in enumerate(lines):
                        if line.strip().startswith("OPENAI_API_KEY="):
                            lines[i] = f'OPENAI_API_KEY="{api_key}"\n'  # Ensure newline
                            key_exists = True
                            break

                    # If key doesn't exist, append it
                    if not key_exists:
                        # Add newline before the key if the last line doesn't end with one
                        if lines and not lines[-1].endswith("\n"):
                            lines.append("\n")
                        lines.append(f'OPENAI_API_KEY="{api_key}"\n')

                    # Write back to file
                    with open(env_path, "w") as env_file:
                        env_file.writelines(lines)
                else:
                    # Create new file
                    with open(env_path, "w") as env_file:
                        env_file.write(f'OPENAI_API_KEY="{api_key}"\n')

                print(f"✅ API key saved to {env_path}")
            except Exception as e:
                print(f"⚠️  Error saving API key to .env file: {e}")
                print("Your API key will be used for this session only.")
        else:
            # Exit
            print("Exiting... Please set up your API key and try again.")
            sys.exit(0)
