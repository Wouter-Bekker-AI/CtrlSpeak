import json
from pathlib import Path

import pytest

from utils import config_paths
from utils.transcription_backend import ApiTranscriptionClient, BackendConfig, save_backend_config, get_backend_config
from utils.ui_state import cleanup_result_label

pytestmark = pytest.mark.core_headless


def test_cleanup_preference_round_trip_and_validation():
    config=save_backend_config(backend='api',api_url='http://gateway:8765',api_token=None,
                              feedback_capture_method='disabled',gpu_cleanup_enabled=True)
    assert config.gpu_cleanup_enabled is True
    assert get_backend_config({}).gpu_cleanup_enabled is True
    assert json.loads(config_paths.get_config_file_path().read_text())['gpu_cleanup_enabled'] is True
    # An unrelated settings save must not silently erase the cleanup preference.
    assert save_backend_config(backend='api',api_url='http://gateway:8765',api_token=None,
                               feedback_capture_method='disabled').gpu_cleanup_enabled is True
    with pytest.raises(ValueError):
        save_backend_config(backend='api',api_url='http://gateway:8765',api_token=None,
                            feedback_capture_method='disabled',gpu_cleanup_enabled='false')


@pytest.mark.parametrize('enabled,provider,status,expected', [
    (True,'ubuntu-gpu-large-v3-turbo','applied','Cleaned.'),
    (False,'ubuntu-gpu-large-v3-turbo','applied','corrected'),
    (True,'openai-gpt-transcribe','applied','corrected'),
    (True,'ubuntu-gpu-large-v3-turbo','timeout','corrected'),
    (True,'ubuntu-gpu-large-v3-turbo','unchanged','corrected'),
])
def test_selection_and_request(tmp_path,enabled,provider,status,expected):
    class Response:
        status_code=200
        def json(self):
            return {'id':'tx', 'raw_text':'raw', 'text':'corrected', 'language':'en',
                    'provider_used':provider, 'normalized_text':'Cleaned.',
                    'normalization':{'requested':True,'applied':status=='applied','status':status,
                                     'model':'S1-mini by Superwhisper','device':'cuda','location':'ubuntu-worker'}}
    class Session:
        def post(self,url,**kwargs):
            assert (kwargs.get('data',{}).get('cleanup')=='true') is enabled
            return Response()
    audio=tmp_path/'audio.wav'
    audio.write_bytes(b'RIFF')
    config=BackendConfig('api','http://gateway',None,'disabled',gpu_cleanup_enabled=enabled)
    result=ApiTranscriptionClient(config,session=Session()).transcribe(audio)
    assert result.text==expected
    assert result.corrected_text=='corrected'
    assert result.raw_text=='raw'


def test_ui_status_and_checkbox_contract():
    assert cleanup_result_label({'client_cleanup_requested':False})=='Cleanup off'
    assert cleanup_result_label({'client_cleanup_requested':True,'client_cleanup_applied':True})=='S1 GPU cleanup applied'
    assert 'secret' not in cleanup_result_label({'client_cleanup_requested':True,'normalization':{'status':'secret'}})
    source=(Path(__file__).parents[2]/'utils/gui.py').read_text('utf-8')
    assert 'Use S1 Mini cleanup when using the Ubuntu GPU' in source
    assert 'gpu_cleanup_enabled=self.gpu_cleanup_var.get()' in source
    visible=(Path(__file__).parents[2]/'utils/midnight_signal_ui.py').read_text('utf-8')
    assert 'Use S1 Mini cleanup when using the Ubuntu GPU' in visible
    assert 'variable=self.gpu_cleanup_var' in visible
    assert 'ms_notebook.select(self.ms_pages[' not in visible
    assert 'getattr(active, "ms_page_tabs", {})' in source
