import os
import warnings

import joblib
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QHBoxLayout, QPushButton, \
    QVBoxLayout, QWidget, QLabel, QFileDialog, QGroupBox, QComboBox, QCheckBox, QSpinBox, \
    QDoubleSpinBox, QFormLayout, QScrollArea
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from DataManagement.data_manager import DataManager
# custom classes
from GuiModules.HelperClasses.user_dialog_module import msg_dlg
from GuiModules.HelperClasses.nan_values_handler_ui import NaValuesHandler
from MLModules.pipeline_manager import PipelineManager


class PipelineManagerUI(QWidget):
    def __init__(self):
        super().__init__()
        self.__initialise_class_variables()
        self.__initial_ui_setup()

    def __initialise_class_variables(self):
        self.__dm: DataManager = DataManager.get_instance()
        self.__df = None  # current dataframe
        self.__features = None  # dataframe after target variable is dropped
        self.__target = None  # references target column
        self.__pipeline_wgts = []  # references for pipeline widgets displayed currently

    def __initial_ui_setup(self):
        """Initialises QT GUI widgets for the Pipelines component"""
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_widget = QWidget()
        self.scroll_area_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.scroll_area)
        #btn for resetting the model creation process
        self.reset_steps_btn = QPushButton('Reset steps')
        self.reset_steps_btn.setDisabled(True)
        self.reset_steps_btn.clicked.connect(self.__reset_steps)
        self.scroll_layout.addWidget(self.reset_steps_btn)
        self.scroll_layout.addWidget(self.__create_dataset_widget())
        self.setLayout(self.main_layout)

        warnings.filterwarnings("ignore", category=ConvergenceWarning)

    def __create_dataset_widget(self)->QGroupBox:
        """Creates widgets with controls that allow the user to select a dataset"""
        self.ds_wgt = QGroupBox()
        self.ds_wgt.setMaximumHeight(150)
        self.ds_wgt.setTitle('Data selection')
        ds_lt = QHBoxLayout()
        ds_main_lt = QVBoxLayout()
        self.ds_wgt.setLayout(ds_main_lt)
        #choice widgets
        current_btn = QPushButton('Use current data')
        load_btn = QPushButton('Load dataset...')
        self.df_next_btn = QPushButton('Next')
        self.df_info_lbl = QLabel()
        self.df_info_lbl.setVisible(False)
        self.df_info_lbl.setStyleSheet("QLabel { color : green; }")

        #add widgets to layout
        ds_lt.addWidget(current_btn)
        ds_lt.addWidget(load_btn)
        ds_lt.addStretch()
        ds_lt.addWidget(self.df_next_btn)
        ds_main_lt.addLayout(ds_lt)
        ds_main_lt.addWidget(self.df_info_lbl)
        #connect actions
        current_btn.clicked.connect(self.__current_btn_action)
        load_btn.clicked.connect(self.__load_btn_action)
        self.df_next_btn.clicked.connect(self.__create_null_handler)
        return self.ds_wgt

    def __create_null_handler(self):
        if self.__df is not None:
            self.nan_handler = NaValuesHandler(self.__df)
            self.nan_handler.setMaximumHeight(260)
            self.scroll_layout.addWidget(self.nan_handler)
            self.nan_handler.next_btn.clicked.connect(self.__create_target_card)
            self.__pipeline_wgts.append(self.nan_handler)
            self.df_next_btn.setDisabled(True)
            self.reset_steps_btn.setEnabled(True)

    def __create_target_card(self):
        if not self.nan_handler.data_is_treated():
            return
        self.__df = self.nan_handler.get_data()
        if self.__df is not None:
            self.target_card = TargetSelectorCard(self.__df.columns)
            self.target_card.setMaximumHeight(200)
            self.target_card.next_btn.clicked.connect(self.__create_transformations_card)
            self.__pipeline_wgts.append(self.target_card)
            self.scroll_layout.addWidget(self.target_card)
            self.nan_handler.next_btn.setDisabled(True)

    def __create_transformations_card(self):
        y_col = self.target_card.get_target_col_name()
        self.__target = self.__df[y_col]  # set targeted column y
        self.__features = self.__df.drop(y_col, axis=1)  # set independent variables X
        self.trs = PipelineTransformerCard(data=self.__features, target=self.__target)
        self.trs.setMaximumHeight(200)
        self.trs.next_btn.clicked.connect(self.__create_model_card)
        self.__pipeline_wgts.append(self.trs)
        self.scroll_layout.addWidget(self.trs)

    def __create_model_card(self):
        self.trs.next_btn.setDisabled(True)
        self.m_card = ModelSelectorCard()
        self.m_card.setMaximumHeight(280)
        self.__pipeline_wgts.append(self.m_card)
        self.scroll_layout.addWidget(self.m_card)
        self.m_card.create_model_btn.clicked.connect(self.__create_ml_model)

    def __reset_steps(self):
        msg = "This action will reset all steps, and any trained model will be deleted. Continue?"
        if not msg_dlg(self,msg=msg,type='warn', title=self.windowTitle()):
            return
        for wgt in list(self.__pipeline_wgts):
            self.scroll_layout.removeWidget(wgt)
            self.__pipeline_wgts.remove(wgt)
            wgt.deleteLater()
        self.df_next_btn.setEnabled(True)
        self.reset_steps_btn.setEnabled(False)

    def __split_data(self):
        return PipelineManager().train_test_split(x=self.__features,
                                                  y=self.__target,
                                                  test_size=self.m_card.get_test_size())

    def __create_ml_model(self):
        try:
            if self.__target is None:
                return
            self.pipeline_mngr = PipelineManager()
            X_train = y_train = X_test = y_test = None
            if self.trs.transform_target_checked():
                self.__target = LabelEncoder().fit_transform(self.__target)
            #split if user chose to train_test
            if self.m_card.train_test_is_selected():
                X_train, X_test, y_train, y_test = self.__split_data()
            else:
                X_train = self.__features
                y_train = self.__target
            #determine if there are object columns, and the type of transformer selected by the user
            cat_col_names = X_train.select_dtypes(include=object).columns.tolist()
            transformer_name = None
            if len(cat_col_names) > 0:
                transformer_name = self.trs.get_selected_transformer()
            #get numerical columns and selected scaler, if any, else don't scale
            num_col_names = X_train.select_dtypes(include='number').columns.tolist()
            scaler_name = self.trs.get_selected_scaler()
            #create column transformer
            col_transformer:ColumnTransformer = self.pipeline_mngr.generate_column_transformer(cat_cols=cat_col_names,
                                                                        num_cols=num_col_names,
                                                                        cat_transformer=transformer_name,
                                                                        num_scaler=scaler_name)
            #create pipeline object
            if not self.m_card.grid_search_is_selected():
                pipeline = self.pipeline_mngr.generate_pipeline(transformer=col_transformer,
                                                           model=self.m_card.get_selected_model(),
                                                           params=self.m_card.get_model_params())
                pipeline.fit(X=X_train, y=y_train)
                self.trained_card = TrainedModelCard(pipeline=pipeline, y_test=y_test, X_test=X_test)
                self.__dm.set_current_model(pipeline) #set current model for DataManager object
                self.trained_card.show()
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FitFailedWarning)
                    pipeline = self.pipeline_mngr.generate_pipeline(transformer=col_transformer,
                                                               model=self.m_card.get_selected_model(),
                                                               )
                    params = self.m_card.get_param_grid()
                    gs = self.pipeline_mngr.grid_search_cv(pipeline=pipeline, param_grid=params, x=X_train, y=y_train)
                    self.trained_card = TrainedModelCard(pipeline=gs.best_estimator_,
                                                         gridsearch_object=gs, X_test=X_test, y_test=y_test)
                    self.__dm.set_current_model(gs.best_estimator_) #set current model to DataManager object
        except Exception as e:
            print(e)

    def __create_gridsearchcv_model(self, pipeline:Pipeline, X, y):
        pass

    def __apply_btn_action(self):
        pass

    def __current_btn_action(self):
        dm = DataManager.get_instance()
        if dm is not None and dm.has_data():
            self.__df = dm.get_data().copy()
            self.df_info_lbl.setVisible(True)
            self.df_info_lbl.setText(f"File selected: Current file --> {dm.shape()}")

    def __load_btn_action(self):
        try:
            fname = self.__load_dataset()[0]
            _n, file_extension = os.path.splitext(fname)
            if file_extension.lower() == '.csv':
                self.__df = pd.read_csv(fname)
            elif file_extension.lower() == '.tsv':
                self.__df = pd.read_csv(fname, sep='\t')
            elif file_extension.lower() == 'xlsx' or file_extension.lower() == 'xlsx':
                self.__df = pd.read_excel(fname)
            elif file_extension.lower() == '.json':
                self.__df = pd.read_json(fname)
            self.df_info_lbl.setVisible(True)
            self.df_info_lbl.setText(f"File selected: {fname}")

        except Exception as e:
            self.err_dlg = msg_dlg(self,f'Critical error {e}',type='error', title=self.windowTitle())

    def __load_dataset(self):
        return QFileDialog.getOpenFileName(self,
                                           'Open file', 'c:\\',
                                           "Doc files (*.csv *.json *.tsv *.xlsx *.xlx)")

#helper classes for PipelineManagerUI
class TargetSelectorCard(QGroupBox):
    def __init__(self, columns:list):
        super().__init__()
        self.columns = columns
        self.setTitle('Target selection')
        self.__initial_setup()

    def __initial_setup(self):
        lbl = QLabel('Features:')
        self.list_wgt = QComboBox()
        self.list_wgt.addItems(self.columns)
        self.next_btn = QPushButton('Next')
        #add widgets to layout
        lt = QHBoxLayout()
        lt.addWidget(lbl)
        lt.addWidget(self.list_wgt)
        lt.addStretch()
        lt.addWidget(self.next_btn)
        self.setLayout(lt)
        #connect btn action
        self.next_btn.clicked.connect(self.next_btn_action)

    def next_btn_action(self):
        self.next_btn.setDisabled(True)

    def get_target_col_name(self):
        return self.list_wgt.currentText()

class PipelineTransformerCard(QGroupBox):
    def __init__(self, data:DataFrame, target:pd.Series):
        super().__init__()
        self.data = data
        self.target = target

        self.__initial_ui_setup()

    def __initial_ui_setup(self):
        lt = QHBoxLayout()
        self.setLayout(lt)
        self.scale_wgt = self.__create_scalers_card()
        self.transform_wgt = self.__create_transformers_card()
        self.next_btn = QPushButton('Next')
        lt.addWidget(self.scale_wgt)
        lt.addWidget(self.transform_wgt)
        lt.addStretch()
        lt.addWidget(self.next_btn)

    def __create_scalers_card(self)->QGroupBox:
        group = QGroupBox()
        group.setTitle('Scaling')
        grp_lt = QVBoxLayout()
        self.scaler_list = QComboBox()
        self.scaler_list.addItems(['No scaling','Normal scaler (min-max)', 'Standard scaler (mean:0, std:1)'])
        self.scale_target = QCheckBox('Include targeted variable')
        #add widgets to layout
        grp_lt.addWidget(self.scaler_list)
        group.setLayout(grp_lt)
        return group

    def __create_transformers_card(self)->QGroupBox:
        group = QGroupBox()
        group.setTitle('Encoding')
        grp_lt = QVBoxLayout()
        self.transformer_list = QComboBox()
        self.transformer_list.addItems(['Label encoder', 'One-hot encoder'])
        self.transform_target_chk = QCheckBox('Encode targeted variable')
        # add widgets to layout
        grp_lt.addWidget(self.transformer_list)
        grp_lt.addWidget(self.transform_target_chk)
        group.setLayout(grp_lt)

        #if __target is object dtype, force transform
        if self.target.dtype == object:
            self.transform_target_chk.setChecked(True)
            self.transform_target_chk.setDisabled(True)
        return group

    def transform_target_checked(self):
        return self.transform_target_chk.isChecked() == True

    def get_selected_transformer(self):
        return self.transformer_list.currentText()

    def get_selected_scaler(self):
        selected_scaler = self.scaler_list.currentText()
        return selected_scaler if selected_scaler != 'No scaling' else None

class ModelSelectorCard(QGroupBox):
    def __init__(self):
        super().__init__()
        self.selected_model:str = ''
        self.__grid_search=False
        self.__initial_ui_setup()

    def __initial_ui_setup(self):
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignVCenter)
        self.setLayout(self.layout)
        #create-model button
        self.create_model_btn = QPushButton('Create model')
        self.create_model_btn.setDisabled(True)
        #model selection widgets
        self.layout.addWidget(self.__create_model_selection_wgts())
        self.layout.addWidget(self.__log_reg_param_wgt())
        self.layout.addWidget(self.__neur_net_param_wgt())
        self.layout.addWidget(self.__elast_net_param_wgt())
        self.layout.addWidget(self.create_model_btn)

    def __create_model_selection_wgts(self)->QWidget:
        self.category = QComboBox()
        self.category.addItems(['Prediction', 'Classification'])
        self.class_models = QComboBox()
        self.class_models.addItems(['','Logistic regression', 'Neural network (MLP Classifier)'])
        self.pred_models = QComboBox()
        self.pred_models.addItems(['','Linear regression', 'Elastic net'])
        self.train_test = QCheckBox('Train/test split (test size)')
        self.test_size = QDoubleSpinBox()
        lt = QHBoxLayout()
        self.group = QGroupBox()
        self.group.setTitle('Model selection')
        self.group.setLayout(lt)
        #add widgets to layout
        lt.addWidget(self.category)
        lt.addWidget(self.class_models)
        lt.addWidget(self.pred_models)
        lt.addWidget(self.train_test)
        lt.addWidget(self.test_size)
        #stylisation & behaviour adjustments
        self.test_size.setRange(10.0,90.0)
        self.test_size.setSingleStep(5)
        self.test_size.setValue(33.3)
        self.class_models.setVisible(False)
        self.test_size.setEnabled(False)
        self.test_size.setSuffix('%')
        #connect actions
        self.train_test.stateChanged.connect(self.__train_check_action)
        self.category.currentIndexChanged.connect(self.__cat_box_changed)
        self.class_models.currentIndexChanged.connect(lambda x: self.__set_selected_model(self.class_models.currentText()))
        self.pred_models.currentIndexChanged.connect(lambda x: self.__set_selected_model(self.pred_models.currentText()))

        return self.group

    def __set_selected_model(self, model_name):
        self.selected_model = model_name
        self.__selection_changed()

    def __train_check_action(self):
        self.test_size.setEnabled(self.train_test.isChecked())

    def __cat_box_changed(self, e):
        self.__grid_search = False
        self.pred_models.setVisible(e==0)
        self.class_models.setVisible(e==1)
        self.__selection_changed()

    def __selection_changed(self):
        self.__grid_search = False
        current_option = self.selected_model
        self.el_param_group.setVisible(current_option=='Elastic net')
        self.log_param_group.setVisible(current_option=='Logistic regression')
        self.nn_param_group.setVisible(current_option=='Neural network (MLP Classifier)')
        if current_option != '':
            self.create_model_btn.setEnabled(True)
        else:
            self.create_model_btn.setDisabled(True)

    def __log_reg_param_wgt(self)->QGroupBox:
        self.log_param_group = QGroupBox()
        self.log_param_group.setVisible(False)
        self.log_param_group.setTitle('Hyperparameter tuning')
        self.grid_search = QCheckBox('Perform grid-search')
        #C param widgets
        str_label = QLabel('Regularization strength (C):')
        self.reg_strength = QComboBox()
        self.reg_strength.addItems(['0.001', '0.01', '1', '10'])
        #penalty param widgets
        pen_lbl = QLabel('Penalty type:')
        self.pen_box = QComboBox()
        self.pen_box.addItems(['None','l1', 'l2'])
        #solver algorithm widgets
        sol_lbl = QLabel('Solver algorithm')
        self.le_info_lbl = QLabel()
        self.le_info_lbl.setVisible(False)
        self.le_info_lbl.setStyleSheet("QLabel { color : red; }")
        self.sol_box = QComboBox()
        self.sol_box.addItems(['newton-cg', 'lbfgs', 'liblinear'])

        def grid_search_status(): #nested method for grid_search checkbox
            self.reg_strength.setDisabled(self.grid_search.isChecked())
            self.pen_box.setDisabled(self.grid_search.isChecked())
            self.sol_box.setDisabled(self.grid_search.isChecked())
            self.__grid_search = self.grid_search.isChecked()

        def validate_params():
            solver = self.sol_box.currentText()
            penalty = self.pen_box.currentText()
            if penalty == 'None':
                self.le_info_lbl.setText("WARNING: penalty=None will ignore the C parameters.")
                self.create_model_btn.setEnabled(True) if self.get_selected_model() != '' else \
                    self.create_model_btn.setEnabled(False)
            elif solver == 'newton-cg' and penalty == 'l1':
                self.le_info_lbl.setVisible(True)
                self.le_info_lbl.setText("Solver newton-cg supports only 'l2' or 'none' penalties.")
                self.create_model_btn.setDisabled(True)
            elif solver == 'lbfgs' and penalty == 'l2':
                self.le_info_lbl.setVisible(True)
                self.le_info_lbl.setText("Solver lbfgs supports only 'l1' or 'none' penalties.")
                self.create_model_btn.setDisabled(True)
            else:
                self.le_info_lbl.setText('')
                self.le_info_lbl.setVisible(False)
                self.create_model_btn.setEnabled(True) if self.get_selected_model() != '' else \
                    self.create_model_btn.setEnabled(False)
        # connect actions
        self.grid_search.stateChanged.connect(grid_search_status)
        self.sol_box.currentIndexChanged.connect(validate_params)
        self.pen_box.currentIndexChanged.connect(validate_params)
        #add items to layout
        main_lt = QVBoxLayout()
        main_lt.addWidget(self.grid_search)
        sec_lt = QHBoxLayout()
        sec_lt.addWidget(str_label)
        sec_lt.addWidget(self.reg_strength)
        sec_lt.addStretch()
        sec_lt.addWidget(pen_lbl)
        sec_lt.addWidget(self.pen_box)
        sec_lt.addStretch()
        sec_lt.addWidget(sol_lbl)
        sec_lt.addWidget(self.sol_box)
        sec_lt.addStretch()
        main_lt.addLayout(sec_lt)
        main_lt.addWidget(self.le_info_lbl)
        self.log_param_group.setLayout(main_lt)
        validate_params()
        return self.log_param_group

    def __neur_net_param_wgt(self)->QGroupBox:
        self.nn_param_group = QGroupBox()
        self.nn_param_group.setVisible(False)
        self.nn_param_group.setTitle('Hyperparameter tuning')
        self.nn_grid_search = QCheckBox('Perform grid-search')
        lt = QHBoxLayout() #main widget layout
        #nn_solvers widgets
        alt = QVBoxLayout()
        alt.setAlignment(Qt.AlignVCenter)
        lbl = QLabel('Solver')
        self.nn_solvers = QComboBox()
        self.nn_solvers.addItems(['lbfgs', 'sgd', 'adam'])
        alt.addWidget(lbl)
        alt.addWidget(self.nn_solvers)
        #activation function wgts
        aclbl = QLabel('Activation function')
        self.act_func = QComboBox()
        self.act_func.addItems(['relu', 'identity', 'logistic'])
        aclt = QVBoxLayout()
        aclt.setAlignment(Qt.AlignVCenter)
        aclt.addWidget(aclbl)
        aclt.addWidget(self.act_func)
        #alpha wgts
        allbl = QLabel('Alpha(L2 strength)')
        self.alphas = QComboBox()
        self.alphas.addItems(['1.e-01', '1.e-02', '1.e-03', '1.e-04',
                              '1.e-05', '1.e-06', '1.e-07', '1.e-08', '1.e-09'])
        allt = QVBoxLayout()
        allt.setAlignment(Qt.AlignVCenter)
        allt.addWidget(allbl)
        allt.addWidget(self.alphas)
        #max iter wgt
        mlbl = QLabel('Max iterations')
        self.miter = QSpinBox()
        self.miter.setRange(100,2000)
        self.miter.setSingleStep(100)
        mlt = QVBoxLayout()
        mlt.addWidget(mlbl)
        mlt.addWidget(self.miter)
        #add widgets to layout/nested layouts
        sec_lt = QHBoxLayout()
        sec_lt.addWidget(self.nn_grid_search)
        sec_lt.addLayout(alt)
        sec_lt.addLayout(aclt)
        third_lt = QHBoxLayout()
        third_lt.addLayout(allt)
        third_lt.addLayout(mlt)
        lt.addLayout(sec_lt)
        lt.addLayout(third_lt)
        self.nn_param_group.setLayout(lt)

        def grid_search_checked(e): # grid search check btn action
            self.nn_solvers.setEnabled(e==0)
            self.act_func.setEnabled(e==0)
            self.alphas.setEnabled(e==0)
            self.miter.setEnabled(e==0)
            self.__grid_search = self.nn_grid_search.isChecked()
        self.nn_grid_search.stateChanged.connect(grid_search_checked)

        return self.nn_param_group

    def __elast_net_param_wgt(self)->QGroupBox:
        self.el_param_group = QGroupBox()
        self.el_param_group.setVisible(False)
        self.el_param_group.setTitle('Hyperparameter tuning')
        self.el_grid_check = QCheckBox('Perform grid-search')
        el_lt = QHBoxLayout() #main widget layout
        el_lt.addWidget(self.el_grid_check)
        #alpha wgts
        al_lbl = QLabel('Alpha (L2 strength)')
        self.al_val = QDoubleSpinBox()
        self.al_val.setRange(0.01, 10)
        self.al_val.setSingleStep(1)
        al_lt = QVBoxLayout()
        al_lt.addWidget(al_lbl)
        al_lt.addWidget(self.al_val)
        #L1 ration wgts
        lo_lbl = QLabel('L1 ratio')
        self.lo_vals = QDoubleSpinBox()
        self.lo_vals.setRange(0,1)
        self.lo_vals.setSingleStep(0.1)
        lo_lt = QVBoxLayout()
        lo_lt.addWidget(lo_lbl)
        lo_lt.addWidget(self.lo_vals)
        #fit intercept checkbox
        self.fit_intercept = QCheckBox('Fit intercept')
        #intrecept & normalise checkboxes layout
        fn_lt = QVBoxLayout()
        fn_lt.addWidget(self.fit_intercept)
        #max iter
        iter_lbl = QLabel('Max iterrations')
        self.el_iter_vals = QSpinBox()
        self.el_iter_vals.setRange(100, 3000)
        self.el_iter_vals.setSingleStep(100)
        mi_lt = QVBoxLayout()
        mi_lt.addWidget(iter_lbl)
        mi_lt.addWidget(self.el_iter_vals)
        #add nested layouts to main wgt layout
        el_lt.addLayout(al_lt)
        el_lt.addLayout(lo_lt)
        el_lt.addLayout(fn_lt)
        el_lt.addLayout(mi_lt)
        self.el_param_group.setLayout(el_lt)

        def grid_checked(e): #btn action for grid chk box
            self.al_val.setDisabled(e==2)
            self.lo_vals.setDisabled(e==2)
            self.fit_intercept.setDisabled(e==2)
            self.el_iter_vals.setDisabled(e==2)
            self.__grid_search =self.el_grid_check.isChecked()

        self.el_grid_check.stateChanged.connect(grid_checked)

        return self.el_param_group

    def get_param_grid(self):
        model_param_grid = {'Logistic regression':{'Logistic regression__C': [0.001, 0.01, 0.1, 1, 10],
                                                   'Logistic regression__penalty': [None, 'l1', 'l2'],
                                                   'Logistic regression__solver': ['newton-cg', 'lbfgs', 'liblinear']
                                                   },
                      'Neural network (MLP Classifier)':{'Neural network (MLP Classifier)__alpha': [0.01, 0.1, 1, 10],
                                                         'Neural network (MLP Classifier)__activation': ['relu', 'identity', 'logistic'],
                                                         'Neural network (MLP Classifier)__solver': ['lbfgs', 'sgd', 'adam'],
                                                         'Neural network (MLP Classifier)__max_iter': [1000, 2000, 3000],
                                                         'Neural network (MLP Classifier)__random_state': [42]
                                                         },
                      'Elastic net':{'Elastic net__alpha': [0.1, 0.5, 1.0],
                                     'Elastic net__l1_ratio': [0.1, 0.5, 0.9],
                                     'Elastic net__fit_intercept': [True, False]
                                     },
                            'Linear regression':None
                            }
        return model_param_grid[self.selected_model]

    def get_model_params(self)->dict:
        """Retrieves the user-inputted model parameters corresponding to the currently selected model.

        Returns a dictionary where the key is the name of the selected model,and the value is a sub-dictionary containing the specific parameters."""
        model_param_grid = {'Logistic regression': {'C': float(self.reg_strength.currentText()),
                                                    'penalty': self.pen_box.currentText() if self.pen_box.currentText() != 'None' else None,
                                                    'solver': self.sol_box.currentText()
                                                    },
                            'Neural network (MLP Classifier)': {'alpha': float(self.alphas.currentText()),
                                                         'activation': self.act_func.currentText(),
                                                         'solver': self.nn_solvers.currentText(),
                                                         'max_iter': int(self.miter.value()),
                                                         'random_state': 0
                                                                },
                            'Elastic net': {'alpha': float(self.al_val.value()),
                                            'l1_ratio': float(self.lo_vals.value()),
                                            'fit_intercept': self.fit_intercept.isChecked()
                                            },
                            'Linear regression': None
                            }
        return model_param_grid[self.selected_model]

    def grid_search_is_selected(self)->bool:
        return self.__grid_search

    def train_test_is_selected(self):
        return self.train_test.isChecked()

    def get_test_size(self):
        return self.test_size.value() / 100

    def get_selected_model(self)->str:
        return self.selected_model

class TrainedModelCard(QWidget):
    def __init__(self, pipeline:Pipeline, gridsearch_object:GridSearchCV=None, y_test=None, X_test=None):
        super().__init__()
        self.pipeline_manager = PipelineManager()
        self.pipeline = pipeline
        self.gridsearch = gridsearch_object
        self.estimator = self.__get_pipeline_estimator()
        self.transformer = self.__get_transformer()
        self.X_test = X_test
        self.y_test = y_test

        self.__initial_ui_setup()
        self.show()

    def __initial_ui_setup(self):
        main_lt = QVBoxLayout()
        self.setLayout(main_lt)
        main_lt.addWidget(self.__pipeline_widget())
        if self.transformer:
            main_lt.addWidget(self.__transformer_widget())
        if self.gridsearch:
            main_lt.addWidget(self.__gridsearch_widget())

        #styling options
        self.setMinimumWidth(750)
        self.setFont(QFont("Consolas", 10))
        self.setStyleSheet("QFormLayout { border: 1px solid black; }")

    def __pipeline_widget(self):
        #create widgets
        layout = QFormLayout()
        name_lbl = QLabel(str(self.model))
        self.export_model_btn = QPushButton('Export model...')
        self.export_model_btn.clicked.connect(lambda l: self.export_btn_clicked('pipeline')) #connect button action
        #add widgets to layout
        est_lbl = QLabel('Estimator model:')
        metr_lbl = QLabel('Metrics:')
        metrics_lbl = QLabel(self.pipeline_manager.get_model_metrics(self.pipeline, self.X_test, self.y_test))
        exp_lbl = QLabel(' ')
        layout.addRow(est_lbl, name_lbl)
        layout.addRow(metr_lbl, metrics_lbl)
        layout.addRow(exp_lbl, self.export_model_btn)
        pipe_wgt = QWidget()
        pipe_wgt.setLayout(layout)
        #styling options
        self.export_model_btn.setMaximumWidth(190)
        font = QFont("Consolas", 10)
        name_lbl.setFont(font)
        est_lbl.setFont(font)
        metr_lbl.setFont(font)
        metrics_lbl.setFont(font)
        metrics_lbl.setWordWrap(True)
        return pipe_wgt

    def __transformer_widget(self):
        layout = QFormLayout()
        trans_contents = self.__column_transformer_to_string(self.transformer)
        trans_lbl = QLabel(trans_contents)
        layout.addRow('Column Transformer:', trans_lbl)
        self.export_transformer_btn = QPushButton('Export transformer...')
        self.export_transformer_btn.clicked.connect(lambda x: self.export_btn_clicked(save_type='transformer'))
        layout.addRow(' ', self.export_transformer_btn)
        trs_wgt = QWidget()
        trs_wgt.setLayout(layout)
        #styling options
        self.export_transformer_btn.setMaximumWidth(240)
        trans_lbl.setWordWrap(True)
        return trs_wgt

    def __gridsearch_widget(self)->QWidget:
        layout = QFormLayout()
        gs_wgt = QWidget()
        gs_wgt.setLayout(layout)
        #arange best parameters vertically to string
        best_params = ""
        for key, value in self.gridsearch.best_params_.items():
            best_params += f"{key}: {value}\n"
        params_lbl = QLabel(best_params)
        layout.addRow('Grid search CV \n optimal parameters:', params_lbl)
        #styling options
        params_lbl.setWordWrap(True)
        return gs_wgt

    def __get_pipeline_estimator(self):
        if self.pipeline:
            for name,estimator in self.pipeline.named_steps.items():
                if isinstance(estimator, BaseEstimator) and not isinstance(estimator, ColumnTransformer):
                    self.model = estimator
                    print(type(estimator))
                    return estimator
            return None

    def __get_transformer(self):
        if self.pipeline:
            for name, step in self.pipeline.named_steps.items():
                print(name, step)
                if isinstance(step, BaseEstimator) and isinstance(step, ColumnTransformer):
                    print(step)
                    self.transformer = step
                    return step
            return None

    def __column_transformer_to_string(self, column_transformer):
        transformer_list = column_transformer.transformers_
        transformer_strings = []
        for name, transformer, cols in transformer_list:
            if callable(transformer):
                transformer_str = transformer.__name__
            else:
                transformer_str = transformer.__class__.__name__

            cols_str = ', '.join(map(str, cols))
            transformer_strings.append(f'<b style="color: blue">\n{name} --> {transformer_str} on: </b>{cols_str}')

        return '\n'.join(transformer_strings)

    def export_btn_clicked(self, save_type = 'pipeline'):
        try:
            # Create a save file dialog
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, f"Save {save_type}", "", "Joblib files (*.joblib)",
                                                       options=options)
            # Save the pipeline to the selected file
            if file_path and self.pipeline and save_type == 'pipeline':
                joblib.dump(self.pipeline, file_path)
                msg_dlg(self,f"The pipeline was successfully saved to:\n {file_path}", 'info', self.windowTitle())
            elif file_path and self.transformer and save_type == 'transformer':
                joblib.dump(self.pipeline, file_path)
                msg_dlg(self, f"Column Transformer object was successfully saved to:\n {file_path}", 'info', self.windowTitle())

        except Exception as e:
            msg_dlg(self,f'Critical error: {e}', 'error', self.windowTitle())





