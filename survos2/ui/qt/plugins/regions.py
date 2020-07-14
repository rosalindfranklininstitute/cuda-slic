

from survos2.ui.qt.plugins.base import register_plugin, Plugin
from survos2.ui.qt.plugins.features import MultiSourceComboBox

from survos2.ui.qt.qtcompat import QtWidgets, QtCore
from survos2.ui.qt.components import *
from survos2.ui.qt.control import Launcher, DataModel


class RegionComboBox(LazyComboBox):

    def __init__(self, full=False, header=(None, 'None'), parent=None):
        self.full = full
        super().__init__(header=header, parent=parent)

    def fill(self):
        params = dict(workspace=True, full=self.full)
        result = Launcher.g.run('regions', 'existing', **params)
        if result:
            self.addCategory('Supervoxels')
            for fid in result:
                if result[fid]['kind'] == 'supervoxels':
                    self.addItem(fid, result[fid]['name'])
            self.addCategory('MegaVoxels')
            for fid in result:
                if result[fid]['kind'] == 'megavoxels':
                    self.addItem(fid, result[fid]['name'])


@register_plugin
class RegionsPlugin(Plugin):

    __icon__ = 'fa.qrcode'
    __pname__ = 'regions'
    __views__ = ['slice_viewer']

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        vbox = VBox(self, spacing=10)

        self(IconButton('fa.plus', 'Add SuperVoxel', accent=True),
             connect=('clicked', self.add_supervoxel))

        self.existing_supervoxels = {}
        self.supervoxel_layout = VBox(margin=0, spacing=5)
        vbox.addLayout(self.supervoxel_layout)

        self(IconButton('fa.plus', 'Add MegaVoxel', accent=True))

    def add_supervoxel(self):
        params = dict(order=1, workspace=True)
        result = Launcher.g.run('regions', 'create', **params)

        if result:
            svid = result['id']
            svname = result['name']
            self._add_supervoxel_widget(svid, svname, True)

    def _add_supervoxel_widget(self, svid, svname, expand=False):
        widget = SupervoxelCard(svid, svname)
        widget.showContent(expand)
        self.supervoxel_layout.addWidget(widget)
        self.existing_supervoxels[svid] = widget
        return widget

    def setup(self):
        params = dict(order=1, workspace=True)
        result = Launcher.g.run('regions', 'existing', **params)
        if result:
            # Remove regions that no longer exist in the server
            for region in list(self.existing_supervoxels.keys()):
                if region not in result:
                    self.existing_supervoxels.pop(region).setParent(None)
            # Populate with new region if any
            for supervoxel in sorted(result):
                if supervoxel in self.existing_supervoxels:
                    continue
                params = result[supervoxel]
                svid = params.pop('id', supervoxel)
                svname = params.pop('name', supervoxel)
                if params.pop('kind', 'unknown') != 'unknown':
                    widget = self._add_supervoxel_widget(svid, svname)
                    widget.update_params(params)
                    self.existing_supervoxels[svid] = widget
                else:
                    logger.warn('+ Skipping loading supervoxel: {}, {}'
                                .format(svid, svname))


class SupervoxelCard(Card):

    def __init__(self, svid, svname, parent=None):
        super().__init__(title=svname, collapsible=True, removable=True,
                         editable=True, parent=parent)
        self.svid = svid
        self.svname = svname

        self.svsource = MultiSourceComboBox()
        self.svsource.setMaximumWidth(250)
        self.svshape = LineEdit3D(parse=int, default=10)
        self.svshape.setMaximumWidth(250)
        self.svspacing = LineEdit3D(parse=float, default=1)
        self.svspacing.setMaximumWidth(250)
        self.svcompactness = LineEdit(parse=float, default=30)
        self.svcompactness.setMaximumWidth(250)
        self.compute_btn = PushButton('Compute')

        self.add_row(HWidgets('Source:', self.svsource, stretch=1))
        self.add_row(HWidgets('Shape:', self.svshape, stretch=1))
        self.add_row(HWidgets('Spacing:', self.svspacing, stretch=1))
        self.add_row(HWidgets('Compactness:', self.svcompactness, stretch=1))
        self.add_row(HWidgets(None, self.compute_btn))

        self.compute_btn.clicked.connect(self.compute_supervoxels)

    def card_deleted(self):
        params = dict(region_id=self.svid, workspace=True)
        result = Launcher.g.run('regions', 'remove', **params)
        if result['done']:
            self.setParent(None)

    def card_title_edited(self, newtitle):
        params = dict(region_id=self.svid, new_name=newtitle, workspace=True)
        result = Launcher.g.run('regions', 'rename', **params)
        return result['done']

    def compute_supervoxels(self):
        src = [DataModel.g.dataset_uri(s) for s in self.svsource.value()]
        dst = DataModel.g.dataset_uri(self.svid, group='regions')
        all_params = dict(
            src=src, dst=dst, compactness=self.svcompactness.value(),
            shape=self.svshape.value(), spacing=self.svspacing.value(),
            modal=True
        )
        Launcher.g.run('regions', 'supervoxels', **all_params)

    def update_params(self, params):
        if 'shape' in params:
            self.svshape.setValue(params['shape'])
        if 'compactness' in params:
            self.svcompactness.setValue(params['compactness'])
        if 'spacing' in params:
            self.svspacing.setValue(params['spacing'])
        if 'source' in params:
            for source in params['source']:
                self.svsource.select(source)
