import json
from pathlib import Path
import unittest

import numpy as np

from app.models.schema import SubtitleRequest, VideoParams
from app.services import video


class TestSubtitleBackgroundSettings(unittest.TestCase):
    def test_subtitle_background_is_disabled_by_default(self):
        """New tasks and the standalone subtitle API should not render a subtitle background unless the user asked for one."""
        video_params = VideoParams(video_subject="default subtitle background")
        subtitle_request = SubtitleRequest(video_script="default subtitle background")

        self.assertFalse(video_params.text_background_color)
        self.assertFalse(subtitle_request.text_background_color)

    def test_all_locales_include_subtitle_background_labels(self):
        """
        After the WebUI adds a subtitle-background switch and color picker, every existing language must include the matching
        translation keys so some locales do not show the English internal key directly.
        """
        i18n_dir = Path(__file__).parent.parent.parent / "webui" / "i18n"
        required_keys = {
            "Enable Subtitle Background",
            "Subtitle Background Color",
            "Subtitle Colors Are Indistinguishable",
            "Subtitle Font Does Not Support Text",
            "No Voice",
        }

        for locale_file in i18n_dir.glob("*.json"):
            with self.subTest(locale=locale_file.name):
                data = json.loads(locale_file.read_text(encoding="utf-8"))
                translations = data.get("Translation", {})
                missing_keys = required_keys - translations.keys()

                self.assertEqual(missing_keys, set())

    def test_video_params_accepts_disabled_and_colored_subtitle_background(self):
        """
        The UI sends False or a color string to the backend depending on the switch. Verify the schema still
        accepts both values so a later dependency or type change does not break the WebUI-to-render contract.
        """
        base_params = {
            "video_subject": "subtitle background smoke",
        }

        disabled_params = VideoParams(
            **base_params,
            text_background_color=False,
        )
        colored_params = VideoParams(
            **base_params,
            text_background_color="#123456",
        )

        self.assertFalse(disabled_params.text_background_color)
        self.assertEqual(colored_params.text_background_color, "#123456")

    def test_visible_text_position_centers_actual_mask_bounds(self):
        """
        A TextClip canvas includes font line height and baseline whitespace, so centering the canvas makes
        the subtitle look low inside the background. A fake mask here simulates "visible text pixels
        in the lower half of the canvas" and verifies the helper recomputes y from the real visible region.
        """

        class FakeMask:
            def get_frame(self, _):
                mask = np.zeros((46, 100), dtype=float)
                mask[12:46, 10:90] = 1.0
                return mask

        class FakeTextClip:
            w = 100
            h = 46
            mask = FakeMask()

        x, y = video._get_visible_center_position(
            FakeTextClip(), container_width=100, container_height=93
        )

        self.assertEqual(x, 0)
        # Visible pixel height is 34px, so in a 93px container there should be about 29px above and below;
        # because the mask top starts at 12px, the TextClip itself needs to move up to 18px.
        self.assertEqual(y, 18)

    def test_detects_indistinguishable_subtitle_colors(self):
        invisible_params = VideoParams(
            video_subject="subtitle color validation",
            text_fore_color="#000000",
            text_background_color="#000000",
            stroke_color="#000000",
            stroke_width=1.5,
        )
        different_outline_params = VideoParams(
            video_subject="subtitle color validation",
            text_fore_color="#000000",
            text_background_color="#000000",
            stroke_color="#FFFFFF",
            stroke_width=1.5,
        )
        background_disabled_params = VideoParams(
            video_subject="subtitle color validation",
            text_fore_color="#000000",
            text_background_color=False,
            stroke_color="#000000",
            stroke_width=1.5,
        )

        self.assertTrue(
            video.subtitle_colors_are_indistinguishable(invisible_params)
        )
        self.assertTrue(
            video.subtitle_colors_are_indistinguishable(different_outline_params)
        )
        self.assertFalse(
            video.subtitle_colors_are_indistinguishable(background_disabled_params)
        )

    def test_detects_font_without_chinese_glyphs(self):
        fonts_dir = (
            Path(__file__).parent.parent.parent / "resource" / "fonts"
        )

        self.assertFalse(
            video.subtitle_font_supports_text(
                str(fonts_dir / "BeVietnamPro-Bold.ttf"), "人工智能改变生活"
            )
        )
        self.assertTrue(
            video.subtitle_font_supports_text(
                str(fonts_dir / "MicrosoftYaHeiBold.ttc"), "人工智能改变生活"
            )
        )
        self.assertTrue(
            video.subtitle_font_supports_text(
                str(fonts_dir / "BeVietnamPro-Bold.ttf"), "Artificial intelligence"
            )
        )

    def test_wrap_text_keeps_closing_punctuation_with_text(self):
        """
        When a long Chinese sentence wraps by character, closing punctuation such as a period must not sit alone on a line, or the subtitle background
        is stretched tall by a single small dot. This reproduces the boundary case of a large-font long Chinese sentence.
        """
        font_path = (
            Path(__file__).parent.parent.parent
            / "resource"
            / "fonts"
            / "MicrosoftYaHeiBold.ttc"
        )

        wrapped_text, _ = video.wrap_text(
            "如果你调整字号，中文笔画也不能被黑色背景遮挡。",
            max_width=1642,
            font=str(font_path),
            fontsize=72,
        )

        self.assertNotIn("\n。", wrapped_text)
        self.assertIn("挡。", wrapped_text)
