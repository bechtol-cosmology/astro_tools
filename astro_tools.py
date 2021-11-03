###useful astro functions I've written/collabed on over the years collected in one place

import numpy as np
from scipy.stats import distributions as dist
from astropy import units as u, cosmology as cos
from astroquery.image_cutouts.first import First
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

####main functionality

class radio:
    ###functions specific to radio astronomy
    def __init__(self):
        ##set up cosmology parameters
        
        return

    
    def estimate_flux(nu_want, nu_known, s_known, a):
        'estimate the flux at a given frequency based on flux at known frequency and spectral index'
        
        flux_want = ((nu_want**a)/(nu_known**a))*s_known
        
        return flux_want


    def spec_idx(s1, s2, nu1, nu2, e_1=0, e_2=0, return_err=False):
        'estimate the spectral index of a source based on two flux measurements at different frequencies -- errors are only based on quad sum of flux errors, first order estimate only'
        
        s1,s2 = np.array(s1), np.array(s2)
    
        srat = s1/s2
        nurat = nu1/nu2
    
        alpha = np.log(srat)/np.log(nurat)
    
        if return_err==True:
            e_1, e_2 = np.array(e_1), np.array(e_2)
            erat = np.abs(srat)*(np.sqrt((e_1/s1)**2 + (e_2/s2)**2))
            e_alpha = np.abs(erat/(srat*np.log(10)))
            spidx = (alpha, e_alpha)
        else:
            spidx = alpha
        
        return spidx


    def radio_luminosity(flux, z, a=-0.7, sunit=u.mJy, lunit='W/Hz',
                         cosmology = cos.FlatLambdaCDM(H0=70, Om0=0.3)):
        'calculate the radio luminosity of a source based on flux, redshift, and assumed spectral index, returns log10 of value'
        
        ##convert to array(if not already)
        flux, z = np.array(flux), np.array(z)
    
        ###calulate luminosity distance in m
        dl = cosmology.luminosity_distance(z)

        ###convert flux to W/Hz/m^2
        flux = (flux*sunit).to(lunit + '/m2')

        ##radio power
        radpower = (flux*4*np.pi*dl**2)/((1+z)**a)
        radpower = radpower.to(lunit)
    
        return(radpower)
    


    def FIRST_cutout(position, size=2*u.arcmin, outname=None):
        ###download FIRST cutout and write to file
        if outname is None:
            astring = position.ra.to_string(unit='hour', sep='', precision=3, pad=True)
            astring = astring[:len(astring)-2]
            dstring = position.dec.to_string(unit='deg', sep='', precision=2, pad=True)
            dstring = dstring[:len(dstring)-3] ##removes decimal!
        
            outname = ('J' + astring + dstring + '_FIRST_cutout.fits')

        hdu = First.get_images(position, image_size=size)
        hdu.writeto(outname)

        return 



class stats:
    ###statistical functionality
    def __init__(self):
        return
    
    
    def summary(data, dp=3, flagval=None):
        'print out summary statistics for a given array; pd.describe for arrays instead of pandas data frames'
        
        n = len(data) ###keeps raw numbers of elements in array inc inf/nan
    
        ###remove nans/infs
        data = data[(np.isfinite(data))]
        
        n_notinf = len(data) ##new N
        n_naninf = n-n_notinf ##number of NaN/infs removed
        
        ###if there's a bad data value you want to exclude from your stats, remove this too
        if flagval is not None:
            data = data[(data!=flagval)]
            n_noflag = len(data)
            n_flag = n_notinf - n_noflag
            flagstring = str(flagval)
        else:
            n_flag = 0
            flagstring = 'na'
        
    
        ###final statement (define here to make length of ascii line)
        nanstate = ' ' + str(n_naninf) + ' non-finite values (not included in stats)'
        flagstate = (' ' + str(n_flag) + ' flagged as bad (value==' + flagstring +
                     '; not included in stats)')
        
        bllen = int(max([len(nanstate), len(flagstate)]))
        
        breakline = '-'*bllen
    
        ###print out stats
        print(breakline)
        print(' Data stats')
        print(' ')
        print(' N:        ', len(data))
        print(' Mean:     ', np.round(np.mean(data), dp))
        print(' Std dev.: ', np.round(np.std(data), dp))
        print(' Min:      ', np.round(np.min(data), dp))
        print(' P25:      ', np.round(np.percentile(data, 25), dp))
        print(' P50:      ', np.round(np.percentile(data, 50), dp))
        print(' P75:      ', np.round(np.percentile(data, 75), dp))
        print(' Max:      ', np.round(np.max(data), dp))
        print('')
        print(nanstate)
        print(flagstate)
        print(breakline)
        
        return
    
    
    def binom_frac(k, n, conf=0.683, nround=4):
        'return fraction and binomial errors based on confidence level'
        
        frac = k/n
        p_low = dist.beta.ppf((1-conf)/2, k+1, n-k+1)
        p_up = dist.beta.ppf(1-(1-conf)/2, k+1, n-k+1)
    
        el = p_low - frac
        eu = p_up - frac
        
        fracpluserrs = (np.round(frac, nround), np.round(el, nround), np.round(eu, nround))
        
        return fracpluserrs


class PanSTARRS:
    ###functions for getting panstarrs data
    ###get images obtained from https://ps1images.stsci.edu/ps1image.html

    def getimages(ra,dec,size=240,filters="grizy"):
    
        """Query ps1filenames.py service to get a list of images
        
            ra, dec = position in degrees
            size = image size in pixels (0.25 arcsec/pixel)
            filters = string with filters to include
            Returns a table with the results
            """
    
        service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
        url = ("{service}?ra={ra}&dec={dec}&size={size}&format=fits"
               "&filters={filters}").format(**locals())
        table = Table.read(url, format='ascii')
        return table


    def geturl(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False):
    
        """Get URL for images in the table
        
            ra, dec = position in degrees
            size = extracted image size in pixels (0.25 arcsec/pixel)
            output_size = output (display) image size in pixels (default = size).
            output_size has no effect for fits format images.
            filters = string with filters to include
            format = data format (options are "jpg", "png" or "fits")
            color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
            Returns a string with the URL
            """
    
        if color and format == "fits":
            raise ValueError("color images are available only for jpg or png formats")
        if format not in ("jpg","png","fits"):
            raise ValueError("format must be one of jpg, png, fits")
        table = PanSTARRS.getimages(ra,dec,size=size,filters=filters)
        url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
               "ra={ra}&dec={dec}&size={size}&format={format}").format(**locals())
        if output_size:
            url = url + "&output_size={}".format(output_size)
        # sort filters from red to blue
        flist = ["yzirg".find(x) for x in table['filter']]
        table = table[np.argsort(flist)]
        if color:
            if len(table) > 3:
                # pick 3 filters
                table = table[[0,len(table)//2,len(table)-1]]
            for i, param in enumerate(["red","green","blue"]):
                url = url + "&{}={}".format(param,table['filename'][i])
        else:
            urlbase = url + "&red="
            url = []
            for filename in table['filename']:
                url.append(urlbase+filename)
        return url








