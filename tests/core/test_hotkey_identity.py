from types import SimpleNamespace
import pytest
from utils.hotkeys import gesture_key_name,key_name

pytestmark=pytest.mark.core_headless


@pytest.mark.parametrize("before,after,vk",[("\x03","c",67),("!","1",49),("@","2",50),("é","e",69)])
def test_windows_identity_ignores_modifier_dependent_character(before,after,vk):
    press,release=SimpleNamespace(char=before,vk=vk),SimpleNamespace(char=after,vk=vk)
    assert gesture_key_name(press,platform_name="win32")==gesture_key_name(release,platform_name="win32")
    assert key_name(press)==before and key_name(release)==after # feedback translation is unchanged


@pytest.mark.parametrize("vk,label",[(17,"ctrl_l"),(162,"ctrl_l"),(163,"ctrl_r"),
    (119,"f8"),(120,"f9"),(121,"f10"),(27,"esc")])
def test_configured_controls_and_generic_ctrl_keep_semantic_names(vk,label):
    assert gesture_key_name(SimpleNamespace(vk=vk),platform_name="win32")==label
    assert gesture_key_name(SimpleNamespace(value=SimpleNamespace(vk=vk)),platform_name="win32")==label


@pytest.mark.parametrize("value",[None,False,-1,256,"67"])
def test_no_untrusted_vk_cast_or_unbounded_identity(value):
    assert gesture_key_name(SimpleNamespace(char="c",vk=value),platform_name="win32")=="c"


def test_x11_character_backend_and_existing_string_tests_remain_unchanged():
    assert gesture_key_name(SimpleNamespace(char="é",vk=69),platform_name="linux")=="é"
    assert gesture_key_name("Key.ctrl",platform_name="win32")=="ctrl_l"
    assert gesture_key_name("Key.ctrl_r",platform_name="win32")=="ctrl_r"
