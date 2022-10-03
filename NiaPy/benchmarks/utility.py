"""Implementation of benchmarks utility function."""

from . import Rastrigin, Rosenbrock, Griewank, \
    Sphere, Ackley, Schwefel, Schwefel221, \
    Schwefel222, Whitley, Alpine1, Alpine2, HappyCat, \
    Ridge, ChungReynolds, Csendes, Pinter, Qing, Quintic, \
    Salomon, SchumerSteiglitz, Step, Step2, Step3, Stepint, SumSquares, StyblinskiTang,\
    f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f_cartpole,f_mountaincar,\
    cec1,cec2,cec3,cec4,cec5,cec6,cec7,cec8,cec9,cec10,cec11,cec12,cec13,cec14,cec15,cec16,cec17,cec18,cec19,cec20,\
    cec21,cec22,cec23,cec24,cec25,cec26,cec27,cec28,cec29,cec30


__all__ = ['Utility']


class Utility:

    def __init__(self):
        self.classes = {
            'ackley': Ackley,
            'alpine1': Alpine1,
            'alpine2': Alpine2,
            'chungReynolds': ChungReynolds,
            'csendes': Csendes,
            'griewank': Griewank,
            'happyCat': HappyCat,
            'pinter': Pinter,
            'quing': Qing,
            'quintic': Quintic,
            'rastrigin': Rastrigin,
            'ridge': Ridge,
            'rosenbrock': Rosenbrock,
            'salomon': Salomon,
            'schumerSteiglitz': SchumerSteiglitz,
            'schwefel': Schwefel,
            'schwefel221': Schwefel221,
            'schwefel222': Schwefel222,
            'sphere': Sphere,
            'step': Step,
            'step2': Step2,
            'step3': Step3,
            'stepint': Stepint,
            'styblinskiTang': StyblinskiTang,
            'sumSquares': SumSquares,
            'whitley': Whitley,
            'f1': f1,
            'f2': f2,
            'f3': f3,
            'f4': f4,
            'f5': f5,
            'f6': f6,
            'f7': f7,
            'f8': f8,
            'f9': f9,
            'f10': f10,
            'f11': f11,
            'f12': f12,
            'f13': f13,
            'f14': f14,
            'f15': f15,
            'f16': f16,
            'f17': f17,
            'f18': f18,
            'f19': f19,
            'f20': f20,
            'f21': f21,
            'f22': f22,
            'f23': f23,
            'f_cartpole':f_cartpole,
            'f_mountaincar':f_mountaincar,
            'cec1': cec1,
            'cec2': cec2,
            'cec3': cec3,
            'cec4': cec4,
            'cec5': cec5,
            'cec6': cec6,
            'cec7': cec7,
            'cec8': cec8,
            'cec9': cec9,
            'cec10': cec10,
            'cec11': cec11,
            'cec12': cec12,
            'cec13': cec13,
            'cec14': cec14,
            'cec15': cec15,
            'cec16': cec16,
            'cec17': cec17,
            'cec18': cec18,
            'cec19': cec19,
            'cec20': cec20,
            'cec21': cec21,
            'cec22': cec22,
            'cec23': cec23,
            'cec24': cec24,
            'cec25': cec25,
            'cec26': cec26,
            'cec27': cec27,
            'cec28': cec28,
            'cec29': cec29,
            'cec30': cec30
        }

    def get_benchmark(self, benchmark):
        if not isinstance(benchmark, ''.__class__):
            return benchmark
        if benchmark in self.classes:
            return self.classes[benchmark]()
        raise TypeError('Passed benchmark is not defined!')

    @classmethod
    def __raiseLowerAndUpperNotDefined(cls):
        raise TypeError('Upper and Lower value must be defined!')
