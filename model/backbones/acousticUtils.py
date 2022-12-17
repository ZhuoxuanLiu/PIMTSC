import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

class MelspectrogramLayer(nn.Module):

    def __init__(
        self,
        n_fft=2048,
        win_length=None,
        hop_length=None,
        window_name=None,
        center=True,
        pad_begin=False,
        sample_rate=22050,
        n_mels=128,
        pow=2.0,
        mel_f_min=0.0,
        mel_f_max=None,
        mel_htk=False,
        mel_norm='slaney',
        return_decibel=True,
        db_amin=1e-5,
        db_ref_value=1.0,
        db_dynamic_range=80.0,
        **kwargs,
    ):
    
        super().__init__(**kwargs)

        if win_length is None:
            win_length = n_fft
        if hop_length is None:
            hop_length = win_length // 4

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window_name = window_name
        self.window_fn = get_window_fn(window_name)
        self.center = center
        self.pow = pow
        self.return_decibel = return_decibel
        self.db_amin = db_amin
        self.db_ref_value = db_ref_value
        self.db_dynamic_range = db_dynamic_range
        
        if self.center:
            self.pad_begin = False
        else:
            self.pad_begin = pad_begin
            
        filterbank_kwargs = {
            'sample_rate': sample_rate,
            'n_freq': n_fft // 2 + 1,
            'n_mels': n_mels,
            'f_min': mel_f_min,
            'f_max': mel_f_max,
            'htk': mel_htk,
            'norm': mel_norm,
        }
        
        self.mel_filter_bank = filterbank_mel(**filterbank_kwargs)
        

    def forward(self, x):
        """
        Compute STFT of the input signal. If the `time` axis is not the last axis of `x`, it should be transposed first.

        Args:
            x (float `Tensor`): batch of audio signals, (batch, ch, time) or (batch, time, ch) based on input_data_format

        Return:
            `frames` is the number of frames, which is `((len_src + (win_length - hop_length) / hop_length) // win_length )` if `pad_end` is `True`.
            `freq` is the number of fft unique bins, which is `n_fft // 2 + 1` (the unique components of the FFT).
        """
        batch_size, input_len, input_dim = x.shape
        waveforms = x.permute(0, 2, 1).reshape(-1, input_len)
        # make sure always (batch, time) from here

        if self.pad_begin:
            waveforms = F.pad(waveforms, (int(self.n_fft - self.hop_length), 0), "constant", 0)
            
        stft = torch.stft(
            input=waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window_fn(self.win_length).to(waveforms.device),
            center=self.center,
            return_complex=True,
        )  # (*, frames, freq)
        
        stftm = torch.pow(torch.abs(stft), self.pow)  # (*, frames, freq) # magnitude spectrogram
        
        output = torch.einsum('...ij, ik -> ...kj', stftm, self.mel_filter_bank.to(stftm.device))

        if self.return_decibel:
            output = magnitude_to_decibel(output, ref_value=self.db_ref_value, 
                                          amin=self.db_amin, dynamic_range=self.db_dynamic_range)
        output = output.reshape(batch_size, input_dim, output.shape[1], output.shape[2]).permute(0, 3, 2, 1)
        return output
    
    
    
def magnitude_to_decibel(x, ref_value=1.0, amin=1e-5, dynamic_range=80.0):
    """A function that converts magnitude to decibel scaling.
    In essence, it runs `10 * log10(x)`, but with some other utility operations.

    Similar to `librosa.power_to_db` with `ref=1.0` and `top_db=dynamic_range`

    Args:
        x (`Tensor`): float tensor. Can be batch or not. Something like magnitude of STFT.
        ref_value (`float`): an input value that would become 0 dB in the result.
            For spectrogram magnitudes, ref_value=1.0 usually make the decibel-scaled output to be around zero
            if the input audio was in [-1, 1].
        amin (`float`): the noise floor of the input. An input that is smaller than `amin`, it's converted to `amin`.
        dynamic_range (`float`): range of the resulting value. E.g., if the maximum magnitude is 30 dB,
            the noise floor of the output would become (30 - dynamic_range) dB

    Returns:
        log_spec (`Tensor`): a decibel-scaled version of `x`.

    Note:
        In many deep learning based application, the input spectrogram magnitudes (e.g., abs(STFT)) are decibel-scaled
        (=logarithmically mapped) for a better performance.

    Example:
        ::

            input_shape = (2048, 1)  # mono signal
            model = Sequential()
            model.add(kapre.Frame(frame_length=1024, hop_length=512, input_shape=input_shape))
            # now the shape is (batch, n_frame=3, frame_length=1024, ch=1)

    """

    if amin is None:
        amin = 1e-5

    amin = torch.tensor(amin).to(dtype=x.dtype)
    ref_value = torch.tensor(ref_value).to(dtype=x.dtype)
    log_spec = 10.0 * torch.log10(torch.maximum(x, amin))
    log_spec = log_spec - 10.0 * torch.log10(torch.maximum(amin, ref_value))

    log_spec = torch.maximum(
        log_spec, torch.max(log_spec.reshape(x.shape[0], -1), dim=1, keepdims=True).values.unsqueeze(-1) - dynamic_range
    )

    return log_spec


    
def filterbank_mel(
    sample_rate, n_freq, n_mels=128, f_min=0.0, f_max=None, htk=False, norm='slaney'
):
    """A wrapper for librosa.filters.mel that additionally does transpose and tensor conversion

    Args:
        sample_rate (`int`): sample rate of the input audio
        n_freq (`int`): number of frequency bins in the input STFT magnitude.
        n_mels (`int`): the number of mel bands
        f_min (`float`): lowest frequency that is going to be included in the mel filterbank (Hertz)
        f_max (`float`): highest frequency that is going to be included in the mel filterbank (Hertz)
        htk (bool): whether to use `htk` formula or not
        norm: The default, 'slaney', would normalize the the mel weights by the width of the mel band.

    Returns:
        (`Tensor`): mel filterbanks. Shape=`(n_freq, n_mels)`
    """
    filterbank = librosa.filters.mel(
        sr=sample_rate,
        n_fft=(n_freq - 1) * 2,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        htk=htk,
        norm=norm,
    )
    return torch.tensor(filterbank.T).float()

    
def get_window_fn(window_name=None):
    """Return a window function given its name.
    This function is used inside layers such as `STFT` to get a window function.

    Args:
        window_name (None or str): name of window function. On Tensorflow 2.3, there are five windows available in
        `tf.signal` (`hamming_window`, `hann_window`, `kaiser_bessel_derived_window`, `kaiser_window`, `vorbis_window`).

    """

    if window_name is None:
        return torch.hann_window

    available_windows = {
        'hamming_window': torch.hamming_window,
        'hann_window': torch.hann_window,
    }
    if hasattr(torch, 'kaiser_window'):
        available_windows['kaiser_window'] = torch.kaiser_window
    if hasattr(torch, 'blackman_window'):
        available_windows['blackman_window'] = torch.blackman_window
    if hasattr(torch, 'bartlett_window'):
        available_windows['bartlett_window'] = torch.bartlett_window

    if window_name not in available_windows:
        raise NotImplementedError(
            'Window name %s is not supported now. Currently, %d windows are'
            'supported - %s'
            % (
                window_name,
                len(available_windows),
                ', '.join([k for k in available_windows.keys()]),
            )
        )

    return available_windows[window_name]



class Normalization2D(nn.Module):

    def __init__(self, int_axis=None, eps=1e-10, **kwargs):

        assert int_axis in (-1, 0, 1, 2, 3), 'invalid int_axis: ' + str(int_axis)
        self.axis = int_axis
        self.eps = eps
        super(Normalization2D, self).__init__(**kwargs)

    def forward(self, x):
        if self.axis == -1:
            mean = torch.mean(x, dim=[3, 2, 1, 0], keepdims=True)
            std = torch.std(x, dim=[3, 2, 1, 0], keepdims=True)
        elif self.axis in (0, 1, 2, 3):
            all_dims = [0, 1, 2, 3]
            del all_dims[self.axis]
            mean = torch.mean(x, axis=all_dims, keepdims=True)
            std = torch.std(x, axis=all_dims, keepdims=True)
        return (x - mean) / (std + self.eps)
    
    
class DropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num):
        """Drop stripes. 

        Args:
          dim: int, dimension along which to drop
          drop_width: int, maximum width of stripes to drop
          stripes_num: int, how many stripes to drop
        """
        super(DropStripes, self).__init__()

        assert dim in [2, 3]    # dim 2: time; dim 3: frequency

        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def forward(self, input):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        assert input.ndimension() == 4

        if self.training is False:
            return input

        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]

            for n in range(batch_size):
                self.transform_slice(input[n], total_width)

            return input


    def transform_slice(self, e, total_width):
        """e: (channels, time_steps, freq_bins)"""

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 2:
                e[:, bgn : bgn + distance, :] = 0
            elif self.dim == 3:
                e[:, :, bgn : bgn + distance] = 0


class SpecAugmentation(nn.Module):
    def __init__(self, time_drop_width, time_stripes_num, freq_drop_width, 
        freq_stripes_num):
        """Spec augmetation. 
        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D. 
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method 
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.

        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
        """

        super(SpecAugmentation, self).__init__()

        self.time_dropper = DropStripes(dim=2, drop_width=time_drop_width, 
            stripes_num=time_stripes_num)

        self.freq_dropper = DropStripes(dim=3, drop_width=freq_drop_width, 
            stripes_num=freq_stripes_num)

    def forward(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        return x