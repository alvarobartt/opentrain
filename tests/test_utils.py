from opentrain.utils import list_fine_tunes


def test_list_fine_tunes():
    fine_tunes = list_fine_tunes(just_succeeded=True)
    assert isinstance(fine_tunes, list)
    assert len(fine_tunes) > 0
    assert isinstance(fine_tunes[0], str)
