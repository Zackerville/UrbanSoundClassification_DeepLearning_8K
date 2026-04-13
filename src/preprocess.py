import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from config import (
    TARGET_SAMPLE_RATE,
    TARGET_NUM_SAMPLES,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    TOP_DB,
    AUDIO_DIR,
)

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)

db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=TOP_DB)


def load_audio(audio_path):
    waveform, sample_rate = sf.read(str(audio_path), always_2d=True)
    waveform = torch.tensor(waveform, dtype=torch.float32).T
    return waveform, sample_rate


def convert_to_mono(waveform):
    if waveform.size(0) == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def resample_audio(waveform, original_sample_rate, target_sample_rate=TARGET_SAMPLE_RATE):
    if original_sample_rate == target_sample_rate:
        return waveform
    resampler = torchaudio.transforms.Resample(
        orig_freq=original_sample_rate,
        new_freq=target_sample_rate,
    )
    return resampler(waveform)


def fix_audio_length(waveform, target_num_samples=TARGET_NUM_SAMPLES):
    current_num_samples = waveform.size(1)

    if current_num_samples == target_num_samples:
        return waveform

    if current_num_samples > target_num_samples:
        return waveform[:, :target_num_samples]

    pad_size = target_num_samples - current_num_samples
    return F.pad(waveform, (0, pad_size))


def waveform_to_log_mel_spectrogram(waveform):
    mel = mel_transform(waveform)
    log_mel = db_transform(mel)
    return log_mel


def normalize_feature(feature):
    mean = feature.mean()
    std = feature.std()
    return (feature - mean) / (std + 1e-6)


def preprocess_waveform(waveform, sample_rate, normalize=True):
    waveform = convert_to_mono(waveform)
    waveform = resample_audio(waveform, sample_rate, TARGET_SAMPLE_RATE)
    waveform = fix_audio_length(waveform, TARGET_NUM_SAMPLES)
    feature = waveform_to_log_mel_spectrogram(waveform)

    if normalize:
        feature = normalize_feature(feature)

    return feature


def preprocess_audio_file(audio_path, normalize=True):
    waveform, sample_rate = load_audio(audio_path)
    feature = preprocess_waveform(waveform, sample_rate, normalize=normalize)
    return feature


def get_preprocessing_info():
    return {
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "target_num_samples": TARGET_NUM_SAMPLES,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "top_db": TOP_DB,
    }


if __name__ == "__main__":
    sample_path = next(AUDIO_DIR.rglob("*.wav"))
    feature = preprocess_audio_file(sample_path)

    print("Sample path:", sample_path)
    print("Feature shape:", feature.shape)
    print("Min:", feature.min().item())
    print("Max:", feature.max().item())
    print("Mean:", feature.mean().item())
    print("Std:", feature.std().item())
    print(get_preprocessing_info())