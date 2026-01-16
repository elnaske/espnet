from argparse import ArgumentParser
from pathlib import Path

import pytest
import torch
import torch.nn.quantized.dynamic as nnqd

from espnet2.bin.gan_codec_inference import AudioCoding, get_parser, main
from espnet2.tasks.gan_codec import GANCodecTask


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


@pytest.fixture()
def config_file(tmp_path: Path):
    # Write default configuration file
    GANCodecTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "gan_codec"),
        ]
    )
    return tmp_path / "gan_codec" / "config.yaml"


@pytest.mark.execution_timeout(5)
def test_AudioCoding(config_file):
    audio_coding = AudioCoding(train_config=config_file)
    wav = torch.rand(1, 16000)
    output_dict = audio_coding(wav)
    assert "resyn_audio" in output_dict
    assert "codes" in output_dict
    assert isinstance(output_dict["resyn_audio"], torch.Tensor)
    assert isinstance(output_dict["codes"], torch.Tensor)


@pytest.mark.execution_timeout(5)
def test_AudioCoding_decode(config_file):
    audio_coding = AudioCoding(train_config=config_file)
    wav = torch.rand(1, 16000)
    encode_dict = audio_coding(wav, encode_only=True)
    assert "codes" in encode_dict
    assert "resyn_audio" not in encode_dict
    assert isinstance(encode_dict["codes"], torch.Tensor)

    decode_dict = audio_coding.decode(encode_dict["codes"])
    assert "resyn_audio" in decode_dict
    assert isinstance(decode_dict["resyn_audio"], torch.Tensor)


@pytest.mark.execution_timeout(5)
@pytest.mark.parametrize("quantize_modules", [["Linear"], ["all"]])
@pytest.mark.parametrize("quantize_dtype", ["qint8", "float16"])
def test_AudioCoding_quantized(config_file, quantize_modules, quantize_dtype):
    audio_coding = AudioCoding(
        train_config=config_file,
        quantize_model=True,
        quantize_modules=quantize_modules,
        quantize_dtype=quantize_dtype,
    )

    assert any(isinstance(m, nnqd.Linear) for m in audio_coding.model.modules())

    # float16 quantization is only applied during matmul, so checking dtype only works for qint8
    if quantize_dtype == "qint8":
        for m in audio_coding.model.modules():
            if isinstance(m, nnqd.Linear):
                assert m.weight().dtype == torch.qint8

    wav = torch.rand(1, 16000)
    output_dict = audio_coding(wav)
    assert "resyn_audio" in output_dict
    assert "codes" in output_dict
    assert isinstance(output_dict["resyn_audio"], torch.Tensor)
    assert isinstance(output_dict["codes"], torch.Tensor)


def test_AudioCoding_quantized_invalid_module_list(config_file):
    with pytest.raises(ValueError):
        AudioCoding(
            train_config=config_file,
            quantize_modules=["all", "Linear"]
        )
