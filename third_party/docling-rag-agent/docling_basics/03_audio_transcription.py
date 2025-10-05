"""
Audio Transcription with Docling + Whisper
==========================================

This script demonstrates audio file transcription using
Docling's ASR (Automatic Speech Recognition) pipeline with
OpenAI's Whisper model.

Features:
- Transcribes audio files (MP3, WAV, M4A, FLAC)
- Uses Whisper Turbo for fast, accurate transcription
- Includes timestamps for temporal reference
- Supports multiple languages (90+)

Prerequisites:
    - FFmpeg must be installed on your system

    Windows (Chocolatey):
        choco install ffmpeg

    Windows (Conda):
        conda install -c conda-forge ffmpeg

    macOS:
        brew install ffmpeg

    Linux:
        apt-get install ffmpeg  # Debian/Ubuntu
        yum install ffmpeg      # RedHat/CentOS

Usage:
    python 03_audio_transcription.py
"""

from docling.document_converter import DocumentConverter, AudioFormatOption
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.datamodel import asr_model_specs
from docling.datamodel.base_models import InputFormat
from docling.pipeline.asr_pipeline import AsrPipeline
from pathlib import Path

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using Whisper ASR."""

    print(f"\n🎙️  Transcribing: {Path(audio_path).name}")
    print("   This may take a moment on first run (downloading Whisper model)...")

    # Configure ASR pipeline with Whisper Turbo
    pipeline_options = AsrPipelineOptions()
    pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

    # Create converter with ASR configuration
    converter = DocumentConverter(
        format_options={
            InputFormat.AUDIO: AudioFormatOption(
                pipeline_cls=AsrPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    # Transcribe
    result = converter.convert(Path(audio_path).resolve())

    # Export to markdown with timestamps
    transcript = result.document.export_to_markdown()

    return transcript

def main():
    print("=" * 60)
    print("Audio Transcription with Docling + Whisper")
    print("=" * 60)

    # Audio file to transcribe
    audio_path = "../documents/Recording1.mp3"

    print(f"\nInput: {audio_path}")

    try:
        # Transcribe
        transcript = transcribe_audio(audio_path)

        # Display results
        print("\n" + "=" * 60)
        print("TRANSCRIPT OUTPUT")
        print("=" * 60)
        print(transcript[:800])  # Show first 800 characters
        if len(transcript) > 800:
            print("\n... (truncated for display)")

        # Save to file
        output_path = "output/output_transcript.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcript)

        print(f"\n✓ Full transcript saved to: {output_path}")
        print(f"✓ Total length: {len(transcript)} characters")

        # Parse timestamp information
        lines = transcript.split('\n')
        timestamp_lines = [line for line in lines if '[time:' in line]
        if timestamp_lines:
            print(f"✓ Found {len(timestamp_lines)} timestamped segments")
            print(f"\nExample timestamp format:")
            print(f"  {timestamp_lines[0][:80]}...")

    except FileNotFoundError:
        print("\n✗ Error: FFmpeg not found!")
        print("\nPlease install FFmpeg:")
        print("  Windows (Chocolatey): choco install ffmpeg")
        print("  Windows (Conda):      conda install -c conda-forge ffmpeg")
        print("  macOS:                brew install ffmpeg")
        print("  Linux:                apt-get install ffmpeg")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure:")
        print("  1. FFmpeg is installed and in PATH")
        print("  2. Audio file exists and is readable")
        print("  3. Audio format is supported (MP3, WAV, M4A, FLAC)")

if __name__ == "__main__":
    main()
