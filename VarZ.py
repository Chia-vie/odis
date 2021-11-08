import numpy as np
import vaex
import matplotlib.pyplot as plt


class ZVar():
    def __init__(self, path, praefix, suffix, path_all_halos):
        self.path = path
        self.praefix = praefix
        self.suffix = suffix
        self.path_all_halos = path_all_halos

        # Read satellite statistics for all halos
        self.all_halos = vaex.from_ascii(self.path_all_halos, names=['halo_num','sat_num','tac','sat_dmass','ratio_dmass','sat_lum'])

    def calc_zvar(self,uniquename):
        '''
        Args:
            uniquename: the unique file identifier, that differs from all other files
        Returns:
        '''
        # Read particle data
        self.particles = vaex.from_ascii(f'{self.path}{self.praefix}{uniquename}{self.suffix}')

        # Convert uniquename into numeric value:
        try:
            halo_id = int(uniquename)
        except:
            halo_id = int(uniquename[1])

        # Read halo data for current specific halo from all_halos dataframe
        self.this_halo = self.all_halos[self.all_halos['halo_num']==halo_id]

        # Calculate Z
        self.particles['z'] = 10**self.particles.feh

        # group by satellite number and calculate the mean and variance of Z and the stellar mass for each satellite
        self.particles_stats = self.particles.groupby(by='nsat').agg({'z': ['mean', 'var'],'mass':'sum'})

        # Loop through all satellites, starting from the highes satnum/accretion time
        satnum = max(self.this_halo.sat_num.values)
        self.halomean = []
        self.halovar = []
        for satnum in self.this_halo.sat_num.values[::-1]:
            # first entry
            if len(self.halomean) == 0:
                new_halo = self.particles[self.particles['nsat'] == satnum]
            # all others
            else:
                new_halo = vaex.concat([new_halo, self.particles[self.particles['nsat'] == satnum]])

            # Calculate statistics on concatenated new halo
            self.halomean.append(float(new_halo.mean(new_halo.z)))
            self.halovar.append(float(new_halo.var(new_halo.z)))

        # stellar masses, alternatively adopt average M/L ratio of 2 and calculate mass from lsat * M/L
        self.reverse_msat_stellar = self.particles_stats.mass.values[::-1]

        # dark matter masses
        self.reverse_msat_dark = self.this_halo.sat_dmass.values[::-1]  # dark matter masses

        # accretion time
        self.reverse_tac = self.this_halo.tac.values[::-1]

        # instantaneous stellar halo mass ration of the merger wrt to the current stellar halo
        self.mu = self.reverse_msat_stellar / np.cumsum(self.reverse_msat_stellar)

        # same as above but for dark matter
        self.mu_dm = self.reverse_msat_dark / np.cumsum(self.reverse_msat_dark)

        # Compute some merger statistics
        s1 = np.sum(self.reverse_msat_stellar) ** 2
        s2 = np.sum(self.reverse_msat_stellar ** 2)

        # Number of Significant Mergers (Cooper et al. 2013 - like an h-index but for mergers)
        nsig = s1 / s2

        # Most massive (relative) merger
        mumax = np.log10(np.max(self.mu[1:]))

        # Most massive (relative) merger in the last 10 Gyr
        mumax10 = np.log10(np.max(self.mu[np.where(self.reverse_tac < 10.)]))

        # Most massive (relative) merger in the last 5 Gyr
        mumax5 = np.log10(np.max(self.mu[np.where(self.reverse_tac < 5.)]))

        # Time of last merger event
        tlast = np.min(self.reverse_tac)

        # Number of mergers with mass ratio > 1/3
        n3 = len(self.reverse_msat_stellar[np.where(self.mu > 0.33)])

        # Number of mergers with mass ratio > 1/10
        n10 = len(self.reverse_msat_stellar[np.where(self.mu > 0.1)])

        # Number of mergers with mass ratio > 1/100
        n100 = len(self.reverse_msat_stellar[np.where(self.mu > 0.01)])

        # Use these three data points to compute a crude accreted mass function slope
        # Fit power-law satellite mass function to get crude estimate of slope
        alpha, b = np.polyfit(np.log10([0.01, 0.1, 1/3]), np.log10([n100, n10, n3]), 1)

        return {'z_mean': self.particles_stats.z_mean.values,
                'z_var': self.particles_stats.z_var.values,
                'cum_z_mean':self.halomean,
                'cum_z_var' : self.halovar,
                'msat_stellar':self.reverse_msat_stellar,
                'msat_dark':self.reverse_msat_dark,
                'tac':self.reverse_tac,
                'mu':self.mu,
                'mudm':self.mu_dm,
                'nsig':nsig,
                'mumax':mumax,
                'mumax10':mumax10,
                'mumax5':mumax5,
                'tlast':tlast,
                'n3':n3,
                'n10':n10,
                'n100':n100,
                'alpha':alpha,
                'b':b
                }

    def plot_zvar(self, stats):
        # calculate allowed range
        xdum = 10 ** (np.linspace(-2.7, -0.5, 300))
        ydg = 10 ** (-0.688970 + 1.88930 * np.log10(xdum))
        ysc = 10 ** (-1.29112 + 2.48155 * np.log10(xdum))

        # make plot
        fig = plt.figure(figsize=[10, 10])
        ax = fig.add_subplot(111)
        ax.plot(np.log10(xdum), np.log10(ydg), '--', c='black')
        ax.plot(np.log10(xdum), np.log10(ysc), '--', c='black')
        ax.scatter(np.log10(stats['z_mean']), np.log10(stats['z_var']), )
        ax.scatter(np.log10(self.halomean), np.log10(self.halovar), marker='s', c=self.reverse_msat_stellar, s=60, alpha=0.5)
        ax.set_ylabel('$\sigma^2$')
        ax.set_xlabel('mean Z')
        ax.set_xlim()
        plt.show()

    def plot_nsat(self, stats):
        fig2 = plt.figure(figsize=[10, 10])
        ax = fig2.add_subplot(111)
        mudum = np.arange(-3, 0, 0.1)
        ax.plot(np.log10([0.01, 0.1, 1. / 3]), np.log10([stats['n100'], ['n10'], ['n3']]), 'o', markersize=50)
        ax.plot(mudum, mudum * stats['alpha'] + stats['b'], '--')
        ax.set_xlabel('logN')
        ax.set_ylabel('logmu')
        ax.text(-1, 2., "alpha = " + str(np.round(stats['alpha'], 2)), fontsize=40)
        plt.show()