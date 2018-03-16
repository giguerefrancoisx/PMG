# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 15:12:17 2018

@author: giguerf
"""
#import matplotlib
import matplotlib.pyplot as plt
#from cycler import cycler
"""
HOW TO SET COLORCYCLE FOR ALL NEW PLOTS:

Basic
plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

Tableau
plt.rcParams['axes.prop_cycle'] =  plt.rcParamsDefault['axes.prop_cycle']

Viridis
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.viridis.colors)

TO SEE AVAILABLE COLORMAPS USE:

plt.colormaps()
https://matplotlib.org/examples/color/colormaps_reference.html

FROM THE MAPLOTLIB DOCUMENTATION:

Matplotlib ships with 4 perceptually uniform color maps which are
the recommended color maps for sequential data:

  =========   ===================================================
  Colormap    Description
  =========   ===================================================
  inferno     perceptually uniform shades of black-red-yellow
  magma       perceptually uniform shades of black-red-white
  plasma      perceptually uniform shades of blue-red-yellow
  viridis     perceptually uniform shades of blue-green-yellow
  =========   ===================================================

The following colormaps are based on the `ColorBrewer
<http://colorbrewer2.org>`_ color specifications and designs developed by
Cynthia Brewer:

ColorBrewer Diverging (luminance is highest at the midpoint, and
decreases towards differently-colored endpoints):

  ========  ===================================
  Colormap  Description
  ========  ===================================
  BrBG      brown, white, blue-green
  PiYG      pink, white, yellow-green
  PRGn      purple, white, green
  PuOr      orange, white, purple
  RdBu      red, white, blue
  RdGy      red, white, gray
  RdYlBu    red, yellow, blue
  RdYlGn    red, yellow, green
  Spectral  red, orange, yellow, green, blue
  ========  ===================================

ColorBrewer Sequential (luminance decreases monotonically):

  ========  ====================================
  Colormap  Description
  ========  ====================================
  Blues     white to dark blue
  BuGn      white, light blue, dark green
  BuPu      white, light blue, dark purple
  GnBu      white, light green, dark blue
  Greens    white to dark green
  Greys     white to black (not linear)
  Oranges   white, orange, dark brown
  OrRd      white, orange, dark red
  PuBu      white, light purple, dark blue
  PuBuGn    white, light purple, dark green
  PuRd      white, light purple, dark red
  Purples   white to dark purple
  RdPu      white, pink, dark purple
  Reds      white to dark red
  YlGn      light yellow, dark green
  YlGnBu    light yellow, light green, dark blue
  YlOrBr    light yellow, orange, dark brown
  YlOrRd    light yellow, orange, dark red
  ========  ====================================

ColorBrewer Qualitative:

(For plotting nominal data, :class:`ListedColormap` is used,
not :class:`LinearSegmentedColormap`.  Different sets of colors are
recommended for different numbers of categories.)

* Accent
* Dark2
* Paired
* Pastel1
* Pastel2
* Set1
* Set2
* Set3

A set of colormaps derived from those of the same name provided
with Matlab are also included:

  =========   =======================================================
  Colormap    Description
  =========   =======================================================
  autumn      sequential linearly-increasing shades of red-orange-yellow
  bone        sequential increasing black-white color map with
              a tinge of blue, to emulate X-ray film
  cool        linearly-decreasing shades of cyan-magenta
  copper      sequential increasing shades of black-copper
  flag        repetitive red-white-blue-black pattern (not cyclic at
              endpoints)
  gray        sequential linearly-increasing black-to-white
              grayscale
  hot         sequential black-red-yellow-white, to emulate blackbody
              radiation from an object at increasing temperatures
  hsv         cyclic red-yellow-green-cyan-blue-magenta-red, formed
              by changing the hue component in the HSV color space
  jet         a spectral map with dark endpoints, blue-cyan-yellow-red;
              based on a fluid-jet simulation by NCSA [#]_
  pink        sequential increasing pastel black-pink-white, meant
              for sepia tone colorization of photographs
  prism       repetitive red-yellow-green-blue-purple-...-green pattern
              (not cyclic at endpoints)
  spring      linearly-increasing shades of magenta-yellow
  summer      sequential linearly-increasing shades of green-yellow
  winter      linearly-increasing shades of blue-green
  =========   =======================================================

A set of palettes from the `Yorick scientific visualisation
package <https://dhmunro.github.io/yorick-doc/>`_, an evolution of
the GIST package, both by David H. Munro are included:

  ============  =======================================================
  Colormap      Description
  ============  =======================================================
  gist_earth    mapmaker's colors from dark blue deep ocean to green
                lowlands to brown highlands to white mountains
  gist_heat     sequential increasing black-red-orange-white, to emulate
                blackbody radiation from an iron bar as it grows hotter
  gist_ncar     pseudo-spectral black-blue-green-yellow-red-purple-white
                colormap from National Center for Atmospheric
                Research [#]_
  gist_rainbow  runs through the colors in spectral order from red to
                violet at full saturation (like *hsv* but not cyclic)
  gist_stern    "Stern special" color table from Interactive Data
                Language software
  ============  =======================================================


Other miscellaneous schemes:

  ============= =======================================================
  Colormap      Description
  ============= =======================================================
  afmhot        sequential black-orange-yellow-white blackbody
                spectrum, commonly used in atomic force microscopy
  brg           blue-red-green
  bwr           diverging blue-white-red
  coolwarm      diverging blue-gray-red, meant to avoid issues with 3D
                shading, color blindness, and ordering of colors [#]_
  CMRmap        "Default colormaps on color images often reproduce to
                confusing grayscale images. The proposed colormap
                maintains an aesthetically pleasing color image that
                automatically reproduces to a monotonic grayscale with
                discrete, quantifiable saturation levels." [#]_
  cubehelix     Unlike most other color schemes cubehelix was designed
                by D.A. Green to be monotonically increasing in terms
                of perceived brightness. Also, when printed on a black
                and white postscript printer, the scheme results in a
                greyscale with monotonically increasing brightness.
                This color scheme is named cubehelix because the r,g,b
                values produced can be visualised as a squashed helix
                around the diagonal in the r,g,b color cube.
  gnuplot       gnuplot's traditional pm3d scheme
                (black-blue-red-yellow)
  gnuplot2      sequential color printable as gray
                (black-blue-violet-yellow-white)
  ocean         green-blue-white
  rainbow       spectral purple-blue-green-yellow-orange-red colormap
                with diverging luminance
  seismic       diverging blue-white-red
  nipy_spectral black-purple-blue-green-yellow-red-white spectrum,
                originally from the Neuroimaging in Python project
  terrain       mapmaker's colors, blue-green-yellow-brown-white,
                originally from IGOR Pro
  ============= =======================================================
  """

#def colorby(data, by='order', values=None):
#
#    if by == 'max':
#        maxlist = (data.max()-data.max().min())/(data.max().max()-data.max().min())
#        keys = maxlist.sort_values(ascending=False).index
#    elif by == 'min':
#        minlist = (data.min()-data.min().min())/(data.min().max()-data.min().min())
#        keys = minlist.sort_values(ascending=True).index
#    elif by == 'order':
#        keys = data.columns
#    else:
#        raise Exception('No other colorby methods implemented yet')
#
#    if values is None:
#        values = plt.cm.viridis
#
#    if isinstance(values, matplotlib.colors.ListedColormap):
#        cmap = values
#        n_steps = len(cmap.colors)//len(keys)
#        if n_steps >= 1:
#            values = cmap.colors[::n_steps]
#        else:
#            values = cmap.colors
#
#    elif isinstance(values, matplotlib.colors.LinearSegmentedColormap):
#        cmap = values
#        n_steps = cmap.N//len(keys)
#        if n_steps >= 1:
#            values = [cmap(n)[:3] for n in range(0, cmap.N, n_steps)]
#        else:
#            values = [cmap(n)[:3] for n in range(cmap.N)]
#
#
#    values = [values[k%len(values)] for k in range(len(keys))]
#    colors = dict(zip(keys, values))
#
#    return colors

if __name__ == '__main__':
    import PMG.COM.data as data
    import PMG.COM.table as tb
    import PMG.COM.plotstyle as style
    THOR = 'P:/AHEC/Data/THOR/'
    chlist = ['11NECKLO00THFOXA']
    time, fulldata = data.import_data(THOR, chlist)
    df = fulldata['11NECKLO00THFOXA'].dropna(axis=1)
    table = tb.get('THOR')
    table = table[table.TYPE.isin(['Frontale/Véhicule'])]
    slips  = table[table.CBL_BELT.isin(['SLIP'])].CIBLE.tolist()
    oks = table[table.CBL_BELT.isin(['OK'])].CIBLE.tolist()
    plt.close('all')
    #%%
    plt.figure()
    colors = style.colordict(df, 'order', plt.cm.YlGnBu)
    for tcn in df.columns[:]:
        plt.plot(df.loc[:,tcn], color=colors[tcn])
    #%%
    df = df.loc[:,slips+oks]
    colors = style.colordict(df, 'max', plt.cm.YlGnBu_r)
    colors = style.colordict(df, 'max', ['tab:blue','tab:red'], 3)
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    for ax, tcns in zip(axs, [slips, oks]):
        for tcn in tcns:
            ax.plot(df.loc[:,tcn], color=colors[tcn])
