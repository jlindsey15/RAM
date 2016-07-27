import numpy as np

initLR = 2e-3
expDecayRate = .99
decayFreq = 200

# calculate learning rate used over time
maxEpoch = 50000
reportFreq = 1000
epochs = range(0, maxEpoch, reportFreq)

nTimesDecay = np.divide(epochs, decayFreq)
lr_overEp = np.multiply(initLR, np.power(expDecayRate, nTimesDecay))

for e in range(0, maxEpoch/reportFreq):
    print epochs[e], lr_overEp[e]


