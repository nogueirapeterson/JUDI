from cached_property import cached_property

from devito.types import SparseTimeFunction
import numpy as np


__all__ = ['PointSource', 'Receiver', 'RickerSource', 'TimeAxis', 'DGaussSource']


class TimeAxis(object):
    """
    Data object to store the TimeAxis. Exactly three of the four key arguments
    must be prescribed. Because of remainder values it is not possible to create
    a TimeAxis that exactly adhears to the inputs therefore start, stop, step
    and num values should be taken from the TimeAxis object rather than relying
    upon the input values.
    The four possible cases are:
    * start is None: start = step*(1 - num) + stop
    * step is None: step = (stop - start)/(num - 1)
    * num is None: num = ceil((stop - start + step)/step) and
    because of remainder stop = step*(num - 1) + start
    * stop is None: stop = step*(num - 1) + start

    Parameters
    ----------
    start : float, optional
        Start of time axis.
    step : float, optional
        Time interval.
    num : int, optional
        Number of values (Note: this is the number of intervals + 1).
        Stop value is reset to correct for remainder.
    stop : float, optional
        End time.
    """
    def __init__(self, start=None, step=None, num=None, stop=None):
        try:
            if start is None:
                start = step*(1 - num) + stop
            elif step is None:
                step = (stop - start)/(num - 1)
            elif num is None:
                num = int(np.ceil((stop - start + step)/step))
                stop = step*(num - 1) + start
            elif stop is None:
                stop = step*(num - 1) + start
            else:
                raise ValueError("Only three of start, step, num and stop may be set")
        except:
            raise ValueError("Three of args start, step, num and stop may be set")

        if not isinstance(num, int):
            raise TypeError("input argument must be of type int")

        self.start = start
        self.stop = stop
        self.step = step
        self.num = num

    def __str__(self):
        return "TimeAxis: start=%g, stop=%g, step=%g, num=%g" % \
               (self.start, self.stop, self.step, self.num)

    def _rebuild(self):
        return TimeAxis(start=self.start, stop=self.stop, num=self.num)

    @cached_property
    def time_values(self):
        return np.linspace(self.start, self.stop, self.num)


class PointSource(SparseTimeFunction):
    """
    Symbolic data object for a set of sparse point sources

    Parameters
    ----------
    name: String
        Name of the symbol representing this source
    grid: Grid
        Grid object defining the computational domain.
    coordinates: Array
        Point coordinates for this source
    data: (Optional) Data
        values to initialise point data
    ntime: Int (Optional)
        Number of timesteps for which to allocate data
    npoint: Int (Optional)
    Number of sparse points represented by this source
    dimension: Dimension (Optional)
         object for representing the number of points in this source
    Note, either the dimensions `ntime` and `npoint` or the fully
    initialised `data` array need to be provided.
    """

    @classmethod
    def __args_setup__(cls, *args, **kwargs):
        try:
            kwargs['nt'] = kwargs['ntime']
        except KeyError:
            kwargs['nt'] = kwargs.get('time').shape[0]

        # nt_dgauss = kwargs.get('ntime', 1)
        # tn_dgauss = kwargs.get('timen', 1)
        # Either `npoint` or `coordinates` must be provided
        npoint = kwargs.get('npoint')
        if npoint is None:
            coordinates = kwargs.get('coordinates', kwargs.get('coordinates_data'))
            if coordinates is None:
                raise TypeError("Need either `npoint` or `coordinates`")
            kwargs['npoint'] = coordinates.shape[0]

        return args, kwargs

    def __init_finalize__(self, *args, **kwargs):
        data = kwargs.pop('data', None)

        kwargs.setdefault('time_order', 2)
        super(PointSource, self).__init_finalize__(*args, **kwargs)

        # If provided, copy initial data into the allocated buffer
        if data is not None:
            self.data[:] = data


Receiver = PointSource


class RickerSource(PointSource):
    """
    Symbolic object that encapsulate a set of sources with a
    pre-defined Ricker wavelet:
    http://subsurfwiki.org/wiki/Ricker_wavelet
    name: Name for the resulting symbol
    grid: :class:`Grid` object defining the computational domain.
    f0: Peak frequency for Ricker wavelet in kHz
    time: Discretized values of time in ms
    """
    @classmethod
    def __args_setup__(cls, *args, **kwargs):
        kwargs.setdefault('npoint', 1)

        return super(RickerSource, cls).__args_setup__(*args, **kwargs)

    def __init_finalize__(self, *args, **kwargs):
        super(RickerSource, self).__init_finalize__(*args, **kwargs)

        self.f0 = kwargs.get('f0')
        self.a = kwargs.get('a')
        self.t0 = kwargs.get('t0')
        for p in range(kwargs['npoint']):
            self.data[:, p] = self.wavelet(kwargs.get('time'))

    def wavelet(self, timev):
        t0 = self.t0 or 1 / self.f0
        a = self.a or 1
        r = (np.pi * self.f0 * (timev - t0))
        return a * (1-2.*r**2)*np.exp(-r**2)

class DGaussSource(PointSource):
    """
    Symbolic object that encapsulate a set of sources with a
    pre-defined Ricker wavelet:
    http://subsurfwiki.org/wiki/Ricker_wavelet
    name: Name for the resulting symbol
    grid: :class:`Grid` object defining the computational domain.
    f0: Peak frequency for Ricker wavelet in kHz
    time: Discretized values of time in ms
    """
    @classmethod
    def __args_setup__(cls, *args, **kwargs):
        kwargs.setdefault('npoint', 1)

        return super(DGaussSource, cls).__args_setup__(*args, **kwargs)

    def __init_finalize__(self, *args, **kwargs):
        super(DGaussSource, self).__init_finalize__(*args, **kwargs)

        self.f0 = kwargs.get('f0')
        self.a = kwargs.get('a')
        self.t0 = kwargs.get('t0')
        self.tn = kwargs.get('tn')
        for p in range(kwargs['npoint']):
            self.data[:, p] = self.wavelet(kwargs.get('time'))

    def wavelet(self, timev):
        """
        Returns the 1st derivative of a Gaussian wavelet.

        Parameters
        ----------
        f0 : float
            Peak frequency in kHz.
        t : TimeAxis
            Discretized values of time in ms.
        a : float
            Maximum amplitude.
        """
        t0 = self.t0 or 1 / self.f0
        a = self.a or 1
        r = (np.pi * self.f0 * (timev - t0))
        tmp = a * (1-2.*r**2)*np.exp(-r**2)
        # tn = self.tn
        tn = self.tn
        nt = timev.shape[0]
        return self.dgauss_process(tmp, tn, nt)

    def dgauss_process(self, tmp, tn, nt):
        """
        Returns the 1st derivative of a Gaussian wavelet.

        Parameters
        ----------
        f0 : float
            Peak frequency in kHz.
        t : TimeAxis
            Discretized values of time in ms.
        a : float
            Maximum amplitude.
        """
        nf = int(((nt*20)+1)/2 + 1)
        df = 1.0 / tn
        faxis = df * np.arange(nf)
        wavelet = tmp

        # Take the Fourier transform of the source time-function
        R = np.fft.fft(wavelet)
        R = R[0:nf]
        nf = len(R)

        # Compute the Hankel function and multiply by the source spectrum
        U_a = np.zeros((nf), dtype=complex)
        for a in range(1, nf-1):

            U_a[a] = R[a]*(1./(-1j*faxis[a]))

            # Do inverse fft on 0:dt:T and you have analytical solution
        U_t = 1.0/(2.0 * np.pi) * np.real(np.fft.ifft(U_a[:], nt))
        U_t = U_t/np.max(U_t)
        return U_t[0:nt]