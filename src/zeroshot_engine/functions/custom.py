import os
import sys
import getpass


def setup_custom_api_key(api_key_name: str):
    """Set up a custom API key for use with the demo."""
    # Check for API key in environment variables
    api_key = os.environ.get(api_key_name)

    # If not in environment variables, try to read from .env file
    if not api_key:
        env_path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path):
            try:
                # Read .env file
                with open(env_path) as env_file:
                    lines = env_file.readlines()
                    for line in lines:
                        if line.strip().startswith(f"{api_key_name}="):
                            # Extract key from line (handle quotes)
                            key_part = line.strip().split("=", 1)[1]
                            api_key = key_part.strip("\"'")
                            if api_key:
                                os.environ[api_key_name] = api_key
                                print(f"✅ Found {api_key_name} in .env file")
                                break
            except Exception as e:
                print(f"⚠️  Error reading .env file: {e}")

    # If still no key, prompt the user
    if not api_key:
        print(f"\n🔐 API Key Security Information for {api_key_name}:")
        print("--------------------------------")
        print(
            f"• Your {api_key_name} is stored locally only in the .env file or temporary environment variable"
        )
        print("• The key is never transmitted to any third-party servers")
        print(
            "• API calls are made to your specified API endpoint.\n"
        )
        print("• For security, never commit the .env file to public repositories")
        print("• When using .env and git, add the .env to your gitignore")
        print(f"• You can delete your API key from .env at any time")
        print("--------------------------------")

        print(f"\n❌ No {api_key_name} found in environment or .env file.")
        print(
            "⚠️  Note: Using a custom API may incur charges to your account.\n"
        )

        print(f"Please provide your {api_key_name} to continue.")

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
                f"Enter your {api_key_name} (input hidden): "
            ).strip()
            os.environ[api_key_name] = api_key
            print("✅ API key set for this session")

        elif key_choice == "2":
            # Save to .env file
            api_key = getpass.getpass(
                f"Enter your {api_key_name} (input hidden): "
            ).strip()
            os.environ[api_key_name] = api_key

            env_path = os.path.join(os.getcwd(), ".env")
            try:
                # Check if file exists and if so, append or replace
                if os.path.exists(env_path):
                    # Read existing content
                    with open(env_path) as env_file:
                        lines = env_file.readlines()

                    # Check if key already exists
                    key_exists = False
                    for i, line in enumerate(lines):
                        if line.strip().startswith(f"{api_key_name}="):
                            lines[i] = f'{api_key_name}="{api_key}"\n'
                            key_exists = True
                            break

                    # If key doesn't exist, append it
                    if not key_exists:
                        # Add newline before the key if the last line doesn't end with one
                        if lines and not lines[-1].endswith("\n"):
                            lines.append("\n")
                        lines.append(f'{api_key_name}="{api_key}"\n')

                    # Write back to file
                    with open(env_path, "w") as env_file:
                        env_file.writelines(lines)
                else:
                    # Create new file
                    with open(env_path, "w") as env_file:
                        env_file.write(f'{api_key_name}="{api_key}"\n')

                print(f"✅ API key saved to {env_path}")
            except Exception as e:
                print(f"⚠️  Error saving API key to .env file: {e}")
                print("Your API key will be used for this session only.")
        else:
            # Exit
            print("Exiting... Please set up your API key and try again.")
            sys.exit(0)
