import os
import sys
import numpy as np
import pickle
import math

class energyShieldDeltaT:

    def __init__(self,fileName):

        with open(fileName,'rb') as fp:
            self.lut = pickle.load(fp)
        assert len(self.lut) > 0, 'ERROR: LUT is empty'
        self.lut.sort(key=(lambda x: x['offs']))

        self.offsets = np.array([ self.lutObj['offs'] for self.lutObj in self.lut], dtype=np.float64)

        paramList = ['rbar', 'sigma', 'vmax', 'lr', 'deltaFMax', 'betaMax']
        for p in paramList:
            setattr(self, p, self.lut[0][p])
            assert all([self.lutObj[p] == getattr(self, p) for self.lutObj in self.lut]), f'ERROR: lut contains different values for problem parameter \'{p}\'...'

    def rmin(self, xi):
        return self.rbar/(self.sigma*np.cos(xi/2) + 1 - self.sigma)

    def deltaT(self, r=0, xi=0, beta=0, debug=False):
        assert xi >= -np.pi and xi <= np.pi, f'ERROR: xi = {xi} is outside of the range [-pi, pi]'
        assert beta >= -self.betaMax and beta <= self.betaMax, f'ERROR: beta = {beta} is outside of the range [-{self.betaMax}, {self.betaMax}]'

        validOffsets = r - (self.rmin(xi) + self.offsets)
        validOffsetIdxs = np.nonzero( (r - (self.rmin(xi) + self.offsets)) > 0 )[0]
        if debug:
            print(f'DEBUG: offset comparisons {validOffsets}')
            print(f'DEBUG: validOffsets = {validOffsetIdxs}')
        if len(validOffsetIdxs) > 0:
            offsetIdx = validOffsetIdxs[ np.argmin(validOffsets[validOffsetIdxs]) ]
            offset = self.lut[offsetIdx]['offs']
        else:
            return 0
        if debug:
            print(f'DEBUG: using offsetIdx = {offsetIdx} with offset = {offset}; rmin({xi}) = {self.rmin(xi)}')
        xiIncrement = self.lut[offsetIdx]['xiIncrement']
        betaIncrement = self.lut[offsetIdx]['betaIncrement']

        xiIndex = int(np.floor((xi+np.pi)/xiIncrement))
        assert xiIndex >=0 and xiIndex < len(self.lut[offsetIdx]['xiPoints']), 'INTERNAL ERROR: xiIndex out of range'
        betaIndex = int(np.floor((beta+self.betaMax)/betaIncrement))
        assert betaIndex >=0 and betaIndex < len(self.lut[offsetIdx]['betaPoints']), 'INTERNAL ERROR: betaIndex out of range'

        if debug:
            xiLeftEndPt = self.lut[offsetIdx]['xiPoints'][xiIndex]
            betaLeftEndPt = self.lut[offsetIdx]['betaPoints'][betaIndex]
            print(f'DEBUG: xi={xi}; xi interval = [{xiLeftEndPt}, {min(xiLeftEndPt+xiIncrement, np.pi)}]')
            print(f'DEBUG: beta={beta}; beta interval = [{betaLeftEndPt}, {min(betaLeftEndPt+betaIncrement, self.betaMax)}]')

        return self.lut[offsetIdx]['lut'][xiIndex, betaIndex]