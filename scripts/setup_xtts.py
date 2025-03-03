#!/usr/bin/env python3
"""
Script to download and set up XTTS model for text-to-speech.
"""

import os
import sys
import time
from pathlib import Path
import subprocess
import importlib

# Change to project root directory
os.chdir(Path(__file__).parent.parent)

def check_package(package_name):
    """Check if package is installed, if not attempt to install it."""
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed. Attempting to install...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ Successfully installed {package_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to install {package_name}: {e}")
            
            # Special handling for TTS package
            if package_name == "TTS":
                print("Attempting alternative installation methods for TTS...")
                try:
                    # Try installing from GitHub directly
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", 
                        "git+https://github.com/coqui-ai/TTS"
                    ])
                    print(f"✅ Successfully installed {package_name} from GitHub")
                    return True
                except Exception as e2:
                    print(f"❌ Failed to install {package_name} from GitHub: {e2}")
            
            return False

def ensure_dependencies():
    """Ensure all dependencies are installed."""
    # Core packages required for XTTS
    required_packages = ["TTS", "torch", "torchaudio", "numpy", "pydub"]
    
    # Optional packages that improve functionality but aren't essential
    optional_packages = ["psutil"]
    
    all_core_installed = True
    
    print("Checking core dependencies...")
    for package in required_packages:
        if not check_package(package):
            if package == "TTS":
                print("⚠️ TTS package installation failed. This is critical for XTTS functionality.")
                all_core_installed = False
            else:
                print(f"⚠️ Failed to install {package}. Some functionality may be limited.")
    
    print("\nChecking optional dependencies...")
    for package in optional_packages:
        if not check_package(package):
            print(f"Note: Optional package {package} could not be installed.")
            print(f"The system will still work but with reduced functionality.")
    
    return all_core_installed

def check_available_memory():
    """Check if there's enough memory to download and run XTTS."""
    try:
        import psutil
        # XTTS v2 requires at least 8GB RAM, 16GB recommended
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        print(f"Available memory: {available_memory_gb:.2f} GB")
        
        if available_memory_gb < 4:
            print("⚠️ Warning: Low memory detected (< 4GB). XTTS may not work properly.")
            return False
        elif available_memory_gb < 8:
            print("⚠️ Warning: Available memory is less than recommended (< 8GB).")
            print("The model may load but performance could be affected.")
            return True
        else:
            print("✅ Memory check passed")
            return True
    except ImportError:
        # If psutil is not available, skip this check
        print("Note: psutil not available, skipping memory check")
        return True

def download_xtts_model():
    """Download the XTTS model."""
    try:
        print("Importing TTS...")
        from TTS.api import TTS
        
        # Create a reference voice file if none exists
        ref_wav = "samples/reference_voice.wav"
        if not os.path.exists(ref_wav) and not any(f.endswith('.wav') for f in os.listdir('samples')):
            create_reference_voice(ref_wav)
        
        print("⬇️ Downloading XTTS v2 model (this might take a while)...")
        start_time = time.time()
        
        # This will download the model if not already cached
        print("Initializing TTS with XTTS v2 model...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        
        # Generate a short test audio to verify the model works
        print("🎵 Testing model with a short sample...")
        test_output = "samples/test_sample.wav"
        
        # XTTS v2 requires a speaker parameter
        try:
            tts.tts_to_file(
                text="Проверка работы модели", 
                file_path=test_output, 
                language="ru",
                speaker_name="v2_ru"  # Using a default Russian speaker
            )
        except Exception as e:
            print(f"First attempt failed with speaker_name: {e}")
            # Try with speaker_wav
            if os.path.exists(ref_wav):
                print(f"Trying with reference voice file: {ref_wav}")
                tts.tts_to_file(
                    text="Проверка работы модели", 
                    file_path=test_output, 
                    language="ru",
                    speaker_wav=ref_wav
                )
            else:
                # Try with generic speaker parameter
                print("Trying with generic speaker parameter...")
                tts.tts_to_file(
                    text="Проверка работы модели", 
                    file_path=test_output, 
                    language="ru",
                    speaker="random"
                )
        
        if os.path.exists(test_output):
            elapsed_time = time.time() - start_time
            print(f"✅ Model successfully downloaded and tested in {elapsed_time:.2f} seconds")
            print(f"🔊 Test audio saved to {test_output}")
            return True
        else:
            print("❌ Failed to generate test audio")
            return False
            
    except ImportError as e:
        print(f"❌ Error: TTS module not found or could not be imported: {e}")
        print("Please make sure TTS is installed. You can try manually with:")
        print("pip install TTS")
        return False
    except Exception as e:
        print(f"❌ Error downloading or initializing XTTS model: {e}")
        
        # Try alternative speaker options if the first attempt failed
        if "speaker" in str(e).lower() or "speaker_name" in str(e).lower():
            try:
                print("\nTrying alternative speaker parameter...")
                tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                test_output = "test_sample.wav"
                
                # Try with speaker_wav instead of speaker_name
                import os
                # Look for any wav file to use as reference
                wav_files = [f for f in os.listdir('.') if f.endswith('.wav')]
                
                if wav_files:
                    print(f"Using {wav_files[0]} as reference speaker")
                    tts.tts_to_file(
                        text="Проверка работы модели", 
                        file_path=test_output, 
                        language="ru",
                        speaker_wav=wav_files[0]
                    )
                else:
                    # If no wav files available, try with generic speaker name
                    print("No wav files found. Trying with generic speaker name...")
                    tts.tts_to_file(
                        text="Проверка работы модели", 
                        file_path=test_output, 
                        language="ru",
                        speaker="random"
                    )
                
                if os.path.exists(test_output):
                    print("✅ Model successfully tested with alternative speaker parameter")
                    return True
            except Exception as e2:
                print(f"❌ Alternative speaker attempt also failed: {e2}")
        
        return False

def create_reference_voice(file_path="samples/reference_voice.wav"):
    """Create a simple reference voice file using a tone generator."""
    try:
        print("Creating a reference voice file...")
        import numpy as np
        from scipy.io import wavfile
        
        # Generate a simple sine wave as a voice reference (2 seconds)
        sample_rate = 22050
        duration = 2  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Create a tone with some variation to simulate speech
        frequencies = [100, 120, 140, 160]  # Hz - low frequencies similar to speech
        signal = np.zeros_like(t)
        
        for i, freq in enumerate(frequencies):
            segment_duration = duration / len(frequencies)
            start = int(i * segment_duration * sample_rate)
            end = int((i + 1) * segment_duration * sample_rate)
            segment_t = t[start:end] - t[start]
            signal[start:end] = 0.5 * np.sin(2 * np.pi * freq * segment_t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * segment_t))
        
        # Apply fade in and fade out
        fade_duration = int(0.1 * sample_rate)
        fade_in = np.linspace(0, 1, fade_duration)
        fade_out = np.linspace(1, 0, fade_duration)
        
        signal[:fade_duration] *= fade_in
        signal[-fade_duration:] *= fade_out
        
        # Normalize
        signal = signal / np.max(np.abs(signal))
        
        # Convert to int16
        signal = (signal * 32767).astype(np.int16)
        
        # Save as WAV
        wavfile.write(file_path, sample_rate, signal)
        print(f"✅ Reference voice file created: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to create reference voice file: {e}")
        return False

def main():
    """Main function to set up XTTS."""
    print("🚀 Setting up XTTS text-to-speech system...")
    
    if not ensure_dependencies():
        print("\n⚠️ Critical dependencies could not be installed")
        print("XTTS setup cannot continue without the TTS package.")
        print("Please see INSTALLATION.md for manual installation instructions.")
        sys.exit(1)
    
    if not check_available_memory():
        print("\n⚠️ Memory check indicates potential issues")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Setup aborted.")
            sys.exit(1)
    
    if download_xtts_model():
        print("\n✅ XTTS setup completed successfully!")
        print("You can now use the TTS functionality with XTTS in your project.")
        print("To test voice cloning, run: python voice_cloning_demo.py")
    else:
        print("\n❌ XTTS setup failed")
        print("Please see INSTALLATION.md for troubleshooting information.")
        sys.exit(1)

if __name__ == "__main__":
    main()
