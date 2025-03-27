def check_ollama_gpu_support():
    """
    Helper function for setup_ollama():
    Check if Ollama is using GPU acceleration across all operating systems.

    Returns:
        dict: Information about GPU support status
    """
    import os
    import platform
    import subprocess

    import requests

    gpu_info = {
        "gpu_available": False,
        "gpu_used_by_ollama": False,
        "gpu_type": "None",
        "details": {},
    }

    os_type = platform.system()

    # Check system for GPU
    if os_type in ["Windows", "Linux"]:
        try:
            # Check for NVIDIA GPU
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                gpu_info["gpu_available"] = True
                gpu_info["gpu_type"] = "NVIDIA"
                gpu_info["details"]["nvidia_smi"] = result.stdout.strip()

                # More thorough check for Ollama GPU usage
                # Method 1: Check for Ollama processes using GPU memory
                process_check = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,name,used_memory",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )

                # Look for any process that might be Ollama-related
                output_lower = process_check.stdout.lower()
                if any(term in output_lower for term in ["ollama", "llm", "llama"]):
                    gpu_info["gpu_used_by_ollama"] = True
                    gpu_info["details"]["ollama_gpu_usage"] = (
                        process_check.stdout.strip()
                    )

                # Method 2: Check if any process using GPU belongs to Ollama
                if not gpu_info["gpu_used_by_ollama"]:
                    try:
                        # Get all GPU processes
                        gpu_procs = subprocess.run(
                            [
                                "nvidia-smi",
                                "--query-compute-apps=pid",
                                "--format=csv,noheader",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=3,
                        )

                        # For each pid, check if it's related to Ollama
                        for pid_line in gpu_procs.stdout.strip().split("\n"):
                            if not pid_line.strip():
                                continue

                            pid = pid_line.strip()
                            try:
                                # Use ps to get command line for this pid
                                cmd_info = subprocess.run(
                                    ["ps", "-p", pid, "-o", "cmd="],
                                    capture_output=True,
                                    text=True,
                                    timeout=1,
                                )
                                cmd_line = cmd_info.stdout.lower()

                                if any(
                                    term in cmd_line
                                    for term in ["ollama", "llm", "llama"]
                                ):
                                    gpu_info["gpu_used_by_ollama"] = True
                                    gpu_info["details"]["ollama_process"] = cmd_line
                                    break
                            except:
                                pass
                    except:
                        pass

                # Method 3: Check environment variables
                if not gpu_info["gpu_used_by_ollama"]:
                    try:
                        # Find Ollama process
                        ollama_proc = subprocess.run(
                            ["pgrep", "-f", "ollama"],
                            capture_output=True,
                            text=True,
                            timeout=2,
                        )

                        for pid in ollama_proc.stdout.strip().split():
                            try:
                                # Read environment variables from /proc
                                with open(f"/proc/{pid}/environ", "rb") as f:
                                    env_data = f.read().split(b"\0")
                                    env_vars = [
                                        v.decode("utf-8", errors="ignore")
                                        for v in env_data
                                    ]

                                    # Check for CUDA environment variables
                                    if any(
                                        v.startswith(("CUDA_", "OLLAMA_CUDA="))
                                        for v in env_vars
                                    ):
                                        gpu_info["gpu_used_by_ollama"] = True
                                        gpu_info["details"]["cuda_env_vars"] = [
                                            v
                                            for v in env_vars
                                            if v.startswith(("CUDA_", "OLLAMA_CUDA="))
                                        ]
                                        break
                            except:
                                pass
                    except:
                        pass

        except Exception as e:
            gpu_info["details"]["error"] = str(e)

    elif os_type == "Darwin":  # macOS
        # Check for Apple Silicon with Metal support
        if platform.machine() == "arm64":
            gpu_info["gpu_available"] = True
            gpu_info["gpu_type"] = "Apple Silicon (Metal)"
            # For M-series Macs, we assume Metal is being used if available
            gpu_info["gpu_used_by_ollama"] = True

    # Check Ollama API for GPU information
    try:
        # First try the version endpoint
        response = requests.get("http://localhost:11434/api/version", timeout=2)
        if response.status_code == 200:
            api_info = response.json()
            gpu_info["details"]["ollama_api"] = api_info

            # If API response mentions GPU/CUDA
            response_str = str(api_info).lower()
            if any(term in response_str for term in ["cuda", "gpu", "metal"]):
                gpu_info["gpu_used_by_ollama"] = True

        # Then try generating a simple response to check capabilities
        if not gpu_info["gpu_used_by_ollama"]:
            test_data = {
                "model": "mistral:7b",  # Use a known model that supports GPU
                "prompt": "Say hello in one word.",
                "stream": False,
            }
            try:
                generate_response = requests.post(
                    "http://localhost:11434/api/generate", json=test_data, timeout=5
                )
                if generate_response.status_code == 200:
                    resp_json = generate_response.json()
                    # Look for GPU info in full response
                    resp_str = str(resp_json).lower()
                    if any(
                        term in resp_str for term in ["cuda", "gpu", "metal", "nvidia"]
                    ):
                        gpu_info["gpu_used_by_ollama"] = True
            except:
                pass

        # Also try checking if the environment variables for GPU are set
        if not gpu_info["gpu_used_by_ollama"]:
            cuda_env = os.environ.get("OLLAMA_CUDA")
            if cuda_env and cuda_env.lower() in ["1", "true", "yes", "on"]:
                gpu_info["gpu_used_by_ollama"] = True
                gpu_info["details"]["env_vars"] = {"OLLAMA_CUDA": cuda_env}
    except Exception as e:
        gpu_info["details"]["api_error"] = str(e)

    return gpu_info


def check_ollama_installation():
    """
    Helper function for setup_ollama():
    Checks if Ollama is installed and returns version information.

    Returns:
        tuple: (is_installed, version_string)
    """
    import os
    import shutil
    import subprocess

    try:
        # First, check if ollama exists in PATH
        ollama_path = shutil.which("ollama")
        if ollama_path:
            print(f"Found Ollama at: {ollama_path}")

            # Try different version command variations
            version_commands = [
                ["ollama", "version"],
                ["ollama", "--version"],
                ["ollama", "-v"],
                ["ollama", "help"],  # Just to check if ollama runs at all
            ]

            for cmd in version_commands:
                try:
                    version_result = subprocess.run(cmd, capture_output=True, text=True)
                    if version_result.returncode == 0:
                        # If help command worked, we know ollama is working but can't get version
                        if cmd[-1] == "help":
                            return True, "Unknown version (command works)"
                        return True, version_result.stdout.strip()
                except Exception:
                    continue

            # If we get here, ollama exists but we couldn't get version info
            # Let's just confirm it can run basic commands
            try:
                list_result = subprocess.run(
                    ["ollama", "list"], capture_output=True, text=True
                )
                if list_result.returncode == 0:
                    return True, "Unknown version (verified working)"
                print(f"Ollama command found but returned error: {list_result.stderr}")
            except Exception as e:
                print(f"Error running ollama: {e}")

            # If binary exists but commands fail, it's likely installed but might need updating
            return True, "Unknown version (found binary)"

    except FileNotFoundError:
        print("Ollama command not found in PATH.")
        # Check common installation locations
        common_paths = [
            "/usr/local/bin/ollama",
            "/usr/bin/ollama",
            "/opt/ollama/bin/ollama",
            os.path.expanduser("~/.ollama/ollama"),
            os.path.expanduser("~/ollama/ollama"),
        ]

        for path in common_paths:
            if os.path.exists(path):
                print(f"Ollama binary found at: {path} but not in PATH")
                print(
                    f"Try adding this directory to your PATH: export PATH={os.path.dirname(path)}:$PATH"
                )
                return True, "Unknown version (found binary)"

        return False, None


def check_system_requirements(model):
    """Check if system meets the requirements for running the model."""
    import psutil

    # Model memory requirements in GB
    model_memory_requirements = {
        "gemma3:2b": 4,
        "gemma3:4b": 8,
        "gemma3:8b": 12,
        "gemma3:12b": 24,
        "llama3:8b": 16,
        "llama3:70b": 80,
    }

    # Get available system memory in GB
    available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    required_memory = model_memory_requirements.get(
        model, 16
    )  # Default to 16GB if unknown

    if available_memory < required_memory:
        print(
            f"⚠️ WARNING: This model requires ~{required_memory}GB RAM, but only {available_memory:.1f}GB is available."
        )
        print("The model may crash or run very slowly. Consider using a smaller model.")
        return False
    return True


def install_ollama(os_type):
    """
    Helper function for setup_ollama():
    Installs Ollama based on the operating system.

    Args:
        os_type (str): The operating system type (Windows, Darwin, Linux)

    Returns:
        bool: True if installation was successful, False otherwise
    """
    import subprocess

    print("Installing Ollama...")
    try:
        # Platform-specific installation instructions
        if os_type == "Windows":
            print(
                "For Windows, please download and install Ollama from: https://ollama.com/download"
            )
            print("After installation, restart this script.")
            return False

        if os_type == "Darwin":  # macOS
            # Check if Homebrew is installed
            brew_check = subprocess.run(
                ["which", "brew"], capture_output=True, text=True
            )
            if brew_check.returncode == 0:
                print("Installing via Homebrew...")
                subprocess.run(["brew", "install", "ollama"], check=True)
            else:
                # Fall back to shell script
                subprocess.run(
                    ["curl", "-fsSL", "https://ollama.com/install.sh"],
                    stdout=subprocess.PIPE,
                )
                install_process = subprocess.Popen(["sh"], stdin=subprocess.PIPE)
                install_process.communicate(
                    input=b"curl -fsSL https://ollama.com/install.sh | sh"
                )

        else:  # Linux
            subprocess.run(
                ["curl", "-fsSL", "https://ollama.com/install.sh"],
                stdout=subprocess.PIPE,
            )
            install_process = subprocess.Popen(["sh"], stdin=subprocess.PIPE)
            install_process.communicate(
                input=b"curl -fsSL https://ollama.com/install.sh | sh"
            )

        print("Ollama installed successfully!")
        return True

    except Exception as e:
        print(f"Installation failed: {e}")
        print("Please install Ollama manually: https://ollama.com/download")
        return False


def update_ollama(os_type):
    """
    Helper function for setup_ollama():
    Updates Ollama based on the operating system.

    Args:
        os_type (str): The operating system type (Windows, Darwin, Linux)

    Returns:
        bool: True if update was successful, False otherwise
    """
    import subprocess
    import time

    print("Updating Ollama...")
    try:
        if os_type == "Windows":
            print(
                "⚠️  For Windows, please download and install the latest Ollama from: https://ollama.com/download"
            )
            print("After updating, restart this script.")
            return False

        if os_type == "Darwin":  # macOS
            # Check if Homebrew is installed
            brew_check = subprocess.run(
                ["which", "brew"], capture_output=True, text=True
            )
            if brew_check.returncode == 0:
                print("Updating via Homebrew...")
                subprocess.run(["brew", "upgrade", "ollama"], check=True)
            else:
                # Fall back to shell script
                install_process = subprocess.Popen(["sh"], stdin=subprocess.PIPE)
                install_process.communicate(
                    input=b"curl -fsSL https://ollama.com/install.sh | sh"
                )

        else:  # Linux
            install_process = subprocess.Popen(["sh"], stdin=subprocess.PIPE)
            install_process.communicate(
                input=b"curl -fsSL https://ollama.com/install.sh | sh"
            )

        print("Ollama updated successfully!")

        # Restart Ollama service if not Windows
        if os_type != "Windows":
            subprocess.run(["ollama", "serve"], capture_output=True, text=True)
            time.sleep(2)  # Give it a moment to start

        return True

    except Exception as e:
        print(f"Update failed: {e}")
        print("⚠️  Please update Ollama manually: https://ollama.com/download")
        return False


def start_ollama_service(os_type):
    """
    Helper function for setup_ollama():
    Starts the Ollama service based on the operating system.

    Args:
        os_type (str): The operating system type (Windows, Darwin, Linux)

    Returns:
        bool: True if service started or is already running, False otherwise
    """
    import os
    import subprocess
    import time

    try:
        if os_type == "Windows":
            # On Windows, first check if service responds
            import requests

            try:
                requests.get("http://localhost:11434/api/version", timeout=1)
                print("Ollama service is running.")
                return True
            except requests.exceptions.ConnectionError:
                print("Ollama service is not running.")

                # Try to automatically start Ollama
                ollama_path = os.path.expanduser(
                    "~\\AppData\\Local\\Programs\\Ollama\\ollama.exe"
                )
                if os.path.exists(ollama_path):
                    print(f"Found Ollama at: {ollama_path}")
                    print("Attempting to start Ollama automatically...")

                    # Start Ollama as a background process
                    subprocess.Popen(
                        [ollama_path],
                        shell=True,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )

                    # Wait and check if service starts
                    for i in range(5):  # Try 5 times
                        print(f"Waiting for Ollama to start... ({i + 1}/5)")
                        time.sleep(3)
                        try:
                            requests.get(
                                "http://localhost:11434/api/version", timeout=1
                            )
                            print("Ollama service started successfully!")
                            return True
                        except requests.exceptions.ConnectionError:
                            continue

                print("Failed to start Ollama service automatically.")
                print("Please start Ollama from your applications menu or system tray.")
                print("Waiting 10 seconds for manual startup...")
                time.sleep(10)
                return False
        else:
            # macOS and Linux code remains unchanged
            result = subprocess.run(["ollama", "serve"], capture_output=True, text=True)
            if "address already in use" in result.stderr:
                print("Ollama service is already running.\n")
            else:
                print("Starting Ollama service...\n")
                time.sleep(2)  # Give it a moment to start
            return True
    except Exception as e:
        print(f"Error with Ollama service: {e}\n")
        return False


def test_model_compatibility(model):
    """
    Helper function for setup_ollama():
    Tests if the model is compatible with the current Ollama version.

    Args:
        model (str): The model name to test

    Returns:
        bool: True if model exists and is compatible, False otherwise
    """
    import subprocess

    try:
        # First check if the model exists in the list
        model_list = subprocess.run(["ollama", "list"], capture_output=True, text=True)

        if model_list.returncode != 0:
            print(f"Error checking model list: {model_list.stderr}")
            return False

        if model in model_list.stdout:
            print(f"Model '{model}' is in the list.")
            # Since the model exists, let's assume it's compatible unless proven otherwise
            return True
        print(f"Model '{model}' not found. Will need to download.")
        return False

    except Exception as e:
        print(f"Error checking model: {e}")
        return False


def get_model_size_from_ollama(model):
    """
    Helper function for setup_ollama():
    Get model size information from Ollama API.

    Args:
        model (str): Name of the model

    Returns:
        tuple: (size_gb, is_local) where size_gb is the size in GB and is_local indicates if model is already downloaded
    """

    import requests

    # First check local models
    try:
        # Get list of local models
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models_data = response.json()
            if "models" in models_data:
                for model_data in models_data["models"]:
                    if model_data["name"] == model:
                        # Model exists locally, get size in GB
                        size_gb = model_data.get("size", 0) / (1024 * 1024 * 1024)
                        return size_gb, True
    except Exception as e:
        print(f"Error checking local models: {e}")

    # If model not found locally, check registry for available models
    try:
        # Check Ollama's registry info
        response = requests.get(f"https://ollama.com/api/tags/{model}", timeout=5)
        if response.status_code == 200:
            model_info = response.json()
            if "size" in model_info:
                size_gb = model_info["size"] / (1024 * 1024 * 1024)
                return size_gb, False
    except Exception as e:
        print(f"Error checking Ollama registry: {e}")

    # Fallback to estimation method if API calls fail
    return estimate_model_size(model), False


def estimate_model_size(model):
    """
    Helper function for setup_ollama():
    Estimate model size based on model name and known patterns.

    Args:
        model (str): Name of the model

    Returns:
        float: Estimated size in GB
    """
    # Dictionary of known model sizes (in GB)
    known_models = {
        "llama2:7b": 3.8,
        "llama2:13b": 7.3,
        "llama2:70b": 39.0,
        "llama3:8b": 4.7,
        "llama3:70b": 40.0,
        "mistral:7b": 4.1,
        "mixtral:8x7b": 26.0,
        "gemma:2b": 1.4,
        "gemma:7b": 4.8,
        "gemma3:2b": 1.6,
        "gemma3:4b": 2.5,
        "gemma3:8b": 5.0,
        "codellama:7b": 3.8,
        "codellama:13b": 7.3,
        "codellama:34b": 19.1,
        "phi3:14b": 8.2,
        "phi3:mini": 1.6,
        "stable-code:3b": 1.8,
    }

    # First check if it's a known model
    if model in known_models:
        return known_models[model]

    # Extract base model and size info from model name
    base_model = model.split(":")[0] if ":" in model else model
    size_suffix = model.split(":")[1] if ":" in model else ""

    # Make educated guess based on model size suffix
    if size_suffix:
        # Extract number from size suffix (e.g., "7b" -> 7)
        try:
            if "b" in size_suffix.lower():
                size_number = float(size_suffix.lower().replace("b", ""))
                # Rough estimation: ~0.5-0.7 GB per billion parameters
                estimated_size = size_number * 0.6
                return estimated_size
        except ValueError:
            pass

    # Default fallback
    return 4.0  # Default to 4GB if we can't determine


def download_model_with_progress(model):
    """
    Helper function for setup_ollama():
    Download a model using ollama with native progress display in terminal
    and minimal elapsed time display in notebooks.

    Args:
        model (str): The model to download

    Returns:
        bool: True if download successful, False otherwise
    """
    import os
    import subprocess
    import threading
    import time

    try:
        # Check if we're in a Jupyter notebook
        from IPython import get_ipython
        from IPython.display import HTML, clear_output, display

        in_notebook = get_ipython() is not None
    except ImportError:
        in_notebook = False

    print(f"Starting download of model '{model}'...")
    print("This may take several minutes for large models.")

    if in_notebook:
        # For notebooks: extremely simplified display with no progress info
        start_time = time.time()

        # Create a simple status message that only shows elapsed time
        display_id = display(HTML("<p>Download started...</p>"), display_id=True)

        def update_time():
            while True:
                elapsed = time.time() - start_time
                # Extremely minimal display - just text
                display_id.update(
                    HTML(f"""<p>Downloading model... ({elapsed:.1f}s elapsed)</p>
<p><em>⚠️ Note: In Jupyter notebooks, progress display is limited. The download is typically
much faster than the displayed progress suggests.</em></p>""")
                )
                time.sleep(1)

        # Start the timer thread
        timer_thread = threading.Thread(target=update_time)
        timer_thread.daemon = True  # Allow process to exit even if thread is running
        timer_thread.start()

        # Completely suppress all stdout and stderr for Ollama
        with open(os.devnull, "w") as devnull:
            try:
                # Run ollama pull with all output completely discarded
                result = subprocess.run(
                    ["ollama", "pull", model],
                    stdout=devnull,
                    stderr=devnull,
                    check=True,
                )
                success = True
            except subprocess.CalledProcessError:
                success = False

        # Stop the timer updates
        elapsed = time.time() - start_time

        # Final update
        if success:
            display_id.update(
                HTML(
                    f"<p>✅ Model '{model}' downloaded successfully in {elapsed:.1f} seconds</p>"
                )
            )
            print(
                f"✅ Model '{model}' downloaded successfully in {elapsed:.1f} seconds!"
            )
            return True
        display_id.update(
            HTML(
                f"<p>❌ Error downloading model '{model}' after {elapsed:.1f} seconds</p>"
            )
        )
        print(f"❌ Error downloading model '{model}'")
        return False
    # For terminals: Use native terminal output
    try:
        start_time = time.time()
        result = subprocess.run(["ollama", "pull", model], check=True)
        elapsed = time.time() - start_time
        print(f"✅ Model '{model}' downloaded successfully in {elapsed:.1f} seconds!")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Error downloading model '{model}'")
        return False


def check_ollama_updates():
    """
    Helper function for setup_ollama():
    Checks if an Ollama update is available and prompts the user to update if needed.

    Returns:
        bool: True if up to date, False if update available
    """
    import platform
    import re
    import subprocess
    import requests
    import semver

    try:
        # Get current Ollama version
        current_version = None
        os_type = platform.system()

        try:
            # Try to get version from API first
            response = requests.get("http://localhost:11434/api/version", timeout=2)
            if response.status_code == 200:
                current_version = response.json().get("version", "").lstrip("v")
        except requests.exceptions.RequestException:
            # If API fails, try command line
            try:
                if os_type == "Windows":
                    result = subprocess.run(
                        ["ollama", "version"], capture_output=True, text=True
                    )
                    version_match = re.search(
                        r"ollama version (\d+\.\d+\.\d+)", result.stdout
                    )
                    if version_match:
                        current_version = version_match.group(1)
                else:
                    result = subprocess.run(
                        ["ollama", "version"], capture_output=True, text=True
                    )
                    current_version = result.stdout.strip().replace(
                        "ollama version ", ""
                    )
            except Exception:
                return False

        if not current_version:
            return False

        # Get the latest version from GitHub API
        github_response = requests.get(
            "https://api.github.com/repos/ollama/ollama/releases/latest"
        )
        if github_response.status_code == 200:
            latest_version = github_response.json()["tag_name"].lstrip("v")

            # Compare versions
            if semver.compare(latest_version, current_version) > 0:
                return False  # Update available
            return True  # Up to date
    except Exception:
        return False


def setup_ollama(model):
    """
    Main function to handle Ollama setup, including installation, updates, and model preparation.

    Args:
        model (str): The model name to set up

    Returns:
        bool: True if setup was successful, False otherwise
    """
    import platform
    import subprocess

    # Detect operating system
    os_type = platform.system()
    print(f"Detected operating system: {os_type}")

    # Check if Ollama is installed
    is_installed, version = check_ollama_installation()

    # Install Ollama if needed
    if not is_installed:
        print("Ollama is not installed.")
        install_choice = input("Would you like to install Ollama? (yes/no): ").lower()

        if install_choice in ("yes", "y"):
            success = install_ollama(os_type)
            if not success:
                return False
        else:
            print("Ollama is required to use local models.")
            return False
    else:
        # Check for updates
        if check_ollama_updates():
            print(f"✓ Ollama is up to date (version {version})")
        else:
            print(f"⚠️  An update for Ollama is available. Current version: {version}")
            print("📥 Download the latest version at: https://ollama.com/download")

    # Start Ollama service
    service_running = start_ollama_service(os_type)
    if not service_running and os_type == "Windows":
        print("Failed to start Ollama service.")
        return False

    # Pull model if needed
    try:
        print(f"Checking if model '{model}' is available...")
        model_list = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True
        )

        if model not in model_list.stdout:
            print(f"Model '{model}' is not available locally. Downloading...")
            success = download_model_with_progress(model)
            if not success:
                print(f"Failed to download model '{model}'.")
                return False
        else:
            print(f"Model '{model}' is already available.")

        gpu_status = check_ollama_gpu_support()

        # Inform user about GPU usage
        if gpu_status["gpu_available"]:
            if gpu_status["gpu_used_by_ollama"]:
                print(f"🚀  GPU acceleration enabled: {gpu_status['gpu_type']}")
                print("Inference will run significantly faster with GPU acceleration.")
            else:
                print(
                    f"⚠️  GPU detected ({gpu_status['gpu_type']}) but Ollama is NOT using it."
                )
                print("Check Ollama configuration to enable GPU acceleration.")
        else:
            print("ℹ️  Running on CPU only (No GPU detected/configured)")
            print(
                "Inference will be slower. Consider using a system with GPU for better performance."
            )

        return True

    except subprocess.CalledProcessError as e:
        print(f"Error checking or downloading model: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False
