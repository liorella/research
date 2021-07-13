from contextlib import contextmanager
from config import *


@contextmanager
def program_():
    pass


@contextmanager
def for_(variables, iterables):
    pass


@contextmanager
def qrun_():
    pass


@contextmanager
def parallel_():
    pass


@contextmanager
def thread_(name=None):
    pass


@contextmanager
def while_(cond):
    pass


def play(waveform,
         channel,
         duration,
         amplitude=None,
         oscillator=None,
         frame=None,
         sample_rate=None,
         chirp_rate=None,
         condition=True,
         process=None):
    pass


def declare(qua_type, size=None, value=None):
    pass


def assign(variable, expression):
    pass


def random(value):
    pass


def procedure(fun, *args):
    pass


class QuaType:
    pass


fixed = QuaType()


class Integration:
    def full(self, target, iw, qm_input, oscillator):
        pass


integration = Integration()
