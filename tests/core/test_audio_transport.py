from pathlib import Path
import threading
import wave

import av
import numpy as np
import pytest

from server.whisper_transcription.app import audio_transport as transport

pytestmark = pytest.mark.core_headless


def recording(path, channels=1, rate=44100, seconds=1):
    pcm = (np.sin(np.arange(rate * channels * seconds) * .07) * 11000).astype('<i2').tobytes()
    with wave.open(str(path), 'wb') as target:
        target.setnchannels(channels)
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes(pcm)
    return pcm


@pytest.mark.parametrize('channels,rate', [(1, 44100), (2, 48000), (1, 16000)])
def test_lossless_round_trip_and_cleanup(tmp_path, channels, rate):
    original = tmp_path / 'source.wav'
    pcm = recording(original, channels, rate)
    with transport.prepare_audio(original) as prepared:
        assert prepared.outcome == 'lossless_flac'
        assert prepared.upload_bytes < prepared.original_bytes
        assert prepared.content_type == 'audio/flac'
        with av.open(str(prepared.path)) as decoded:
            assert decoded.streams.audio[0].sample_rate == rate
            assert decoded.streams.audio[0].channels == channels
            restored = b''.join(f.to_ndarray().tobytes() for f in decoded.decode(audio=0))
        assert restored == pcm
        np.testing.assert_array_equal(transport.decode_bounded(original), transport.decode_bounded(prepared.path))
        temporary = prepared.path
    assert not temporary.exists()
    assert original.exists()


def test_caller_exception_is_not_swallowed_or_retried(tmp_path):
    original = tmp_path / 'source.wav'
    recording(original)
    with pytest.raises(RuntimeError, match='upload failed'):
        with transport.prepare_audio(original) as prepared:
            temporary = prepared.path
            raise RuntimeError('upload failed')
    assert not temporary.exists()
    assert original.exists()


@pytest.mark.parametrize('head,extension,mime', [(b'OggS', '.ogg', 'audio/ogg'),
                                             (b'fLaC', '.flac', 'audio/flac')])
def test_compressed_bytes_passthrough_despite_wrong_extension(tmp_path, head, extension, mime):
    source = tmp_path / 'wrong.wav'
    content = head + b'opaque-compressed-payload'
    source.write_bytes(content)
    with transport.prepare_audio(source) as prepared:
        assert prepared.path == source
        assert prepared.path.read_bytes() == content
        assert prepared.filename.endswith(extension)
        assert prepared.content_type == mime


def test_original_retained_when_encoder_fails(tmp_path, monkeypatch):
    source = tmp_path / 'source.wav'
    recording(source)
    monkeypatch.setattr(av, 'open', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('encoder failure')))
    with transport.prepare_audio(source) as prepared:
        assert prepared.path == source
        assert prepared.outcome == 'encoder_error_passthrough'


def test_cancellation_and_decoded_limits(tmp_path):
    source = tmp_path / 'source.wav'
    recording(source)
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(InterruptedError):
        with transport.prepare_audio(source, cancelled=cancelled):
            pytest.fail('cancelled preparation should not upload')
    with transport.prepare_audio(source) as prepared:
        with pytest.raises(transport.AudioLimitError):
            transport.decode_bounded(prepared.path, max_pcm_bytes=1024)
        with pytest.raises(transport.AudioLimitError):
            transport.decode_bounded(prepared.path, max_seconds=.01)


def test_invalid_audio_rejected_at_decoder(tmp_path):
    source = tmp_path / 'bad.flac'
    source.write_bytes(b'fLaCbroken')
    with pytest.raises(transport.AudioLimitError):
        transport.decode_bounded(source)


def test_packaged_self_test():
    assert transport.self_test()['bit_perfect'] is True


def test_truncated_flac_is_rejected(tmp_path):
    source = tmp_path / 'source.wav'
    recording(source, seconds=3)
    with transport.prepare_audio(source) as prepared:
        damaged = tmp_path / 'truncated.flac'
        data = prepared.path.read_bytes()
        damaged.write_bytes(data[:len(data) // 2])
        with pytest.raises(transport.AudioLimitError):
            transport.decode_bounded(damaged)


def test_preparation_deadline_and_noisy_not_smaller_retains_source(tmp_path):
    source = tmp_path / 'source.wav'
    recording(source)
    with transport.prepare_audio(source, timeout_seconds=-1) as prepared:
        assert prepared.path == source
    rng = np.random.default_rng(7)
    with wave.open(str(source), 'wb') as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000)
        handle.writeframes(rng.integers(-32768, 32767, 16000, dtype=np.int16).tobytes())
    with transport.prepare_audio(source) as prepared:
        assert prepared.path == source and prepared.outcome == 'not_smaller_passthrough'
