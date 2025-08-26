"""
cmap
颜色
> variable
    ColorLisa
> function
    get
    show
    get_from_list
"""
from .figure import *
from .. import df as ut_df
from ..general import str_step_insert, show_str_arr,update_dict

palettes = json.loads(Path(__file__).parent .joinpath(
    'color/scanpy.plotting.palettes.json').read_text())
palettes.update({k: v.split(',') for k, v in palettes.items()
                 if k.startswith('default')})
def get_color(count):
    palette = None
    if count <= 20:
        palette = palettes['default_20']
    elif count <= 28:
        palette = palettes['default_28']
    elif count <= len(palettes['default_102']):  # 103 colors
        palette = palettes['default_102']
    else:
        raise Exception("[categories too long] {}".format(count))
    return palette[:count]


def get(serise, color_missing_value="lightgray",
        offset=2, filter_offset=True):

    serise = pd.Series(pd.Series(serise).unique())
    has_missing_value = serise.isna().any()
    serise = pd.Series(np.concatenate(
        (['_{}'.format(i) for i in range(offset)], serise.dropna().astype(str))))

    palette = get_color(serise.size)

    colormap = {k: v for k, v in zip(serise, palette)}
    if has_missing_value:
        colormap.update({'nan': color_missing_value})

    if filter_offset:
        colormap = {
            k: v
            for _, (k, v) in zip(
                ~pd.Series(colormap.keys()).str.match('_\\d+'),
                colormap.items())
            if _
        }
    return colormap


def show(color_map,marker='.', size=40, text_x=0.1, kw_scatter=None,
         fontdict=None,axis_off=True,ax=None, return_ax=False):
    """
    Parameters
    ----------
    text_x : float
        控制text的横坐标
        marker 的横坐标为0
        ax的横坐标被锁定为(-0.05, 0.25)
    """
    if ax:
        fig = ax.figure
    else:
        fig, ax = subplots_get_fig_axs()
    if isinstance(size, (int, float)) or (not isinstance(size, Iterable)):
        size = np.repeat(size, len(color_map.keys()))
    if isinstance(marker, str) or (not isinstance(marker, Iterable)):
        marker = np.repeat(marker, len(color_map.keys()))

    fontdict = update_dict(dict(ha='left',va='center'),fontdict)
    kw_scatter = update_dict({},kw_scatter)
    
    for i, ((k, v), m, s) in enumerate(
            zip(color_map.items(), marker, size)):
        ax.scatter(0, len(color_map.keys())-i,
                   label=k, c=v, s=s, marker=m,**kw_scatter)
        ax.text(text_x, len(color_map.keys())-i, k, fontdict=fontdict)
    ax.set_xlim(-0.05, 0.25)
    ax.set_ymargin(.5)
    ax.set_axis_off() if axis_off else None

    return ax if return_ax else fig

def show_cmap_df_with_js(df):
    from IPython.display import display,display_javascript
    display(df.style.set_table_styles([{
        'selector': '.ColorLisa-item',
        'props': 'color:white;background-color:grey;'
    }]).set_td_classes(
        pd.DataFrame(np.full(df.shape, "ColorLisa-item"),
                     index=df.index, columns=df.columns)
    )
    )
    display_javascript(
        """
        $.each($(".ColorLisa-item"),function(i,ele){
        $(ele).css("background-color",$(ele).text())
    })
        """, raw=True
    )


def get_from_list(colors, is_categories=True, name='', **kvarg):
    """
    由颜色列表生成cmap
    Examples
    ----------
    colors = 'darkorange,#279e68,gold,#d62728,lawngreen,#aa40fc,lightseagreen,#8c564b'.split(',')
    display(get_from_list(colors,True))
    display(get_from_list(colors,False))
    """

    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    res = None
    if is_categories:
        res = ListedColormap(colors, name, **kvarg)
    else:
        res = LinearSegmentedColormap.from_list(name, colors, **kvarg)
    return res


# ----------------------------------------
# matplotlib_qualitative_colormaps
# ----------------------------------------


class Qcmap:
    """
    matplotlib_qualitative_colormaps
    >[详见](https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative)

    > function
        get_colors
        get_cmap
        show
    """
    item_size_max = 20
    def __init__(self):
        self.df = pd.read_csv(Path(__file__).parent
                              .joinpath('color/matplotlib_qualitative_colormaps.csv'),index_col=0)
    def get_colors(self,name):
        colors = np.array(self.df.at[name,'colors'].split(','))
        return colors[colors != 'white']
    
    def get_cmap(self,name,keys):
        colors = self.get_colors(name)
        if len(keys) > len(colors):
            print('[Warning][Qcmap][get_cmap] length of keys is greater than colors')
        return {k:v for k,v in zip(keys,colors)}

    def show(self):
        data = self.df['colors'].str.extract(','.join(['(?P<c{}>[^,]+)'.format(i) for i in range(self.item_size_max)]))
        show_cmap_df_with_js(data)
        


# ----------------------------------------
# ColorLisa
# ----------------------------------------



class ColorLisa:
    """
    从ColorLisa获取的颜色,共110项, 每项中有5种颜色
    ColorLisa取材于艺术家的作品

    详见[ColorLisa](https://colorlisa.com/)

    > function
        get_colors
        get_cmap
        show
        show_author
        show_all
        show_all_as_df
    """
    common_color_n_cols = 5

    def __init__(self):
        self.df = pd.read_csv(Path(__file__).parent
                              .joinpath('color/ColorLisa.csv'))\
            .sort_values('author,id'.split(','))\
            .reset_index(drop=True)
        self.df.index = self.df.apply(lambda row:"{author}_{id}".format(**row),axis=1).to_numpy()

    def __get_item(self, author, id=0):
        item = None
        if author in self.df.index:
            item = self.df.loc[author,:]
        if "{}_{}".format(author,id) in self.df.index:
            item = self.df.loc["{}_{}".format(author,id),:]
        assert not item is None, "[Error] can not get item with author={} id={}".format(author,id)
        return item

    def __show_df_cmap(self, df=None, ncols=4, flexible_ncols=True):
        if df is None:
            df = self.df
        if flexible_ncols:
            ncols = min(ncols, df.shape[0])
        nrows = df.shape[0]//ncols + (0 if df.shape[0] % ncols == 0 else 1)

        with plt.rc_context(rc=rc_blank):
            fig, axs = subplots_get_fig_axs(nrows, ncols, ratio_nrows=.7, ratio_ncols=1.4)
            axs = [axs] if nrows*ncols == 1 else axs

        for ax, (i, row) in zip(axs, df.iterrows()):
            show({i:i for i in row['color'].split(',')},ax=ax,text_x=.05)
            ax.set_title(str_step_insert(i, 10),
                fontdict=dict(
                    fontsize=6,ha='center', va='top'))
        return fig

    def help(self):
        print(self.__doc__)

    def get_colors(self,author,id=0):
        return np.array(self.__get_item(author, id)['color'].split(','))
    
    def get_cmap(self, author, id=0,keys=None):
        colors = self.get_colors(author,id)
        keys = [str(i) for i in range(len(colors))] if keys is None else keys
        
        if len(keys) > len(colors):
            print('[Warning][ColorLisa][get_cmap] length of keys is greater than colors')
        return {k:v for k,v in zip(keys,colors)}

    def show(self):
        ut_df.show(self.df)

    def show_author(self):
        show_str_arr(self.df['author'].unique())

    def show_all(self):
        display(self.__show_df_cmap(ncols=6))

    def show_all_as_df(self):
        show_cmap_df_with_js(self.df['color'].str.extract(
            ','.join(['(?P<c{}>[^,]+)'.format(i) for i in range(5)])))

# ----------------------------------------
# ggsci
# ----------------------------------------

class ggsci(Qcmap):
    def __init__(self):
        self.df = pd.read_csv(Path(__file__).parent
                              .joinpath('color/ggsci.csv'),index_col=0)
        self.item_size_max = self.df['length'].max()


# 单例模式
Qcmap = Qcmap()
# ColorLisa = ColorLisa()
ggsci = ggsci()