
from numpy import *
import matplotlib.pyplot as plt


def plot_all(mes, th, ic1, ic2, ss=True, nf=64, sample_f=1.0,
             logss=True, sqrtmes=False, plotf=None):
    '''Plots nxn graphics, with confidence intervals and threshold.
       If ss == True, plots ss in the diagonal.
       Already expects data in power form: abs(x)^2'''
    if logss:
        ss = log(ss)

    if sqrtmes:
        mes = sqrt(mes)
        th = sqrt(th)
        ic1 = sqrt(ic1)
        ic2 = sqrt(ic2)

    x = sample_f*arange(nf)/(2.0*nf)
    n = mes.shape[0]
    plt.figure()

    for i in range(n):
        for j in range(n):
            plt.subplot(n, n, i*n+j+1)
            #over = mes[i,j][mes[i,j]>th[i,j]]
            #overx = x[mes[i,j]>th[i,j]]
            over = mes[i, j]
            overx = x
            if i != j:
                plt.plot(x, th[i, j], 'k--', overx, over, 'r-')  # b  --> r

                # plt.plot(x, th[i, j], 'k--', x, ic1[i, j], 'k:', x, ic2[i, j], 'k:',
                #          overx, over, 'r-')  # b  --> r

            # Complicated code for underthreshold painting
            k = 0
            while(k < nf):
                while(mes[i, j, k] >= th[i, j, k]):
                    k = k+1
                    if (k == nf):
                        break
                if (k == nf):
                    break
                kold = k
                while(mes[i, j, k] < th[i, j, k]):
                    k = k+1
                    if (k == nf):
                        break

                if i != j:
                    plt.plot(x[kold:k], mes[i, j, kold:k], 'g-')  # r -> b

            plt.ylim(-0.05, 1.05)
            if (i < n-1):
                plt.xticks([])
            if (j > 0):
                plt.yticks([])

            if plotf != None:
                plt.xlim([0, plotf])

        if (ss is not None):
            ax = plt.subplot(n, n, i*n+i+1).twinx()
            ax.plot(sample_f*arange(nf)/(2.0*nf), ss[i, i, :], color='b')
            if logss:
                ax.set_ylim(ymin=ss[i, i, :].min(), ymax=ss[i, i, :].max())
            else:
                ax.set_ylim(ymin=0, ymax=ss[i, i, :].max())

            if (i < n-1):
                ax.set_xticks([])

            if plotf != None:
                ax.set_xlim(xmin=0, xmax=plotf)

    plt.show(block=False)


def pdc_plot(pdc2, ss=None, nf=64, sample_f=1.0):
    '''Plots nxn graphics.
       If ss == True, plots ss in the diagonal.
       Expects data in complex form. Does: abs(x)^2 before plotting.'''
    n = pdc2.shape[0]
    # pdc = pdc*pdc.conj() # pdc2 is already a squared pdc estimate
    pdc2 = abs(pdc2)
    plt.figure()

    for i in range(n):
        for j in range(n):
            plt.subplot(n, n, i*n+j+1)
            plt.plot(sample_f*arange(nf)/(2.0*nf), pdc2[i, j, :])
            plt.ylim(-0.05, 1.05)
            if (i < n-1):
                plt.xticks([])
            if (j > 0):
                plt.yticks([])
        if (ss != None):
            ax = plt.subplot(n, n, i*n+i+1).twinx()
            ax.plot(sample_f*arange(nf)/(2.0*nf), ss[i, i, :], color='g')
            ax.set_ylim(ymin=0, ymax=ss[i, i, :].max())
            if (i < n-1):
                ax.set_xticks([])
    plt.show(block=False)
