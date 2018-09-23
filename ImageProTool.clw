; CLW file contains information for the MFC ClassWizard

[General Info]
Version=1
LastClass=CImageProToolView
LastTemplate=generic CWnd
NewFileInclude1=#include "stdafx.h"
NewFileInclude2=#include "ImageProTool.h"
LastPage=0

ClassCount=9
Class1=CImageProToolApp
Class2=CImageProToolDoc
Class3=CImageProToolView
Class4=CMainFrame

ResourceCount=5
Resource1=IDR_MAINFRAME
Resource2=IDR_IMAGEPTYPE
Class5=CChildFrame
Class6=CAboutDlg
Resource3=IDD_ABOUTBOX
Class7=CInputDialog
Resource4=IDD_INPUT
Class8=CHistogram
Class9=CChapter1
Resource5=IDD_HISTOGRAM

[CLS:CImageProToolApp]
Type=0
HeaderFile=ImageProTool.h
ImplementationFile=ImageProTool.cpp
Filter=N

[CLS:CImageProToolDoc]
Type=0
HeaderFile=ImageProToolDoc.h
ImplementationFile=ImageProToolDoc.cpp
Filter=N
BaseClass=CDocument
VirtualFilter=DC
LastObject=ID_FILE_OPEN

[CLS:CImageProToolView]
Type=0
HeaderFile=ImageProToolView.h
ImplementationFile=ImageProToolView.cpp
Filter=C
BaseClass=CScrollView
VirtualFilter=VWC
LastObject=ID_MEANSHIFT


[CLS:CMainFrame]
Type=0
HeaderFile=MainFrm.h
ImplementationFile=MainFrm.cpp
Filter=T


[CLS:CChildFrame]
Type=0
HeaderFile=ChildFrm.h
ImplementationFile=ChildFrm.cpp
Filter=M


[CLS:CAboutDlg]
Type=0
HeaderFile=ImageProTool.cpp
ImplementationFile=ImageProTool.cpp
Filter=D

[DLG:IDD_ABOUTBOX]
Type=1
Class=CAboutDlg
ControlCount=4
Control1=IDC_STATIC,static,1342177283
Control2=IDC_STATIC,static,1342308480
Control3=IDC_STATIC,static,1342308352
Control4=IDOK,button,1342373889

[MNU:IDR_MAINFRAME]
Type=1
Class=CMainFrame
Command1=ID_FILE_NEW
Command2=ID_FILE_OPEN
Command3=ID_FILE_PRINT_SETUP
Command4=ID_FILE_MRU_FILE1
Command5=ID_APP_EXIT
Command6=ID_VIEW_TOOLBAR
Command7=ID_VIEW_STATUS_BAR
Command8=ID_APP_ABOUT
CommandCount=8

[TB:IDR_MAINFRAME]
Type=1
Class=CMainFrame
Command1=ID_FILE_NEW
Command2=ID_FILE_OPEN
Command3=ID_FILE_SAVE
Command4=ID_EDIT_CUT
Command5=ID_EDIT_COPY
Command6=ID_EDIT_PASTE
Command7=ID_FILE_PRINT
Command8=ID_APP_ABOUT
CommandCount=8

[MNU:IDR_IMAGEPTYPE]
Type=1
Class=CImageProToolView
Command1=ID_FILE_NEW
Command2=ID_FILE_OPEN
Command3=ID_FILE_CLOSE
Command4=ID_FILE_SAVE
Command5=ID_FILE_SAVE_AS
Command6=ID_FILE_PRINT
Command7=ID_FILE_PRINT_PREVIEW
Command8=ID_FILE_PRINT_SETUP
Command9=ID_FILE_MRU_FILE1
Command10=ID_APP_EXIT
Command11=ID_RGB_TO_GRAY
Command12=ID_RGB_HSV
Command13=ID_RGB_GRAY
Command14=ID_PIXEL_ADDITION
Command15=ID_SUBTRACTION
Command16=ID_DIVISION
Command17=ID_PIXEL_ADDITION
Command18=ID_LOGICAL_AND
Command19=ID_LOGICAL_AND2
Command20=ID_HISTOGRAM
Command21=ID_HISTO_THRESHOLD
Command22=ID_GLOBAL_THRESHOLDING
Command23=ID_ADAPTIVE_THRESHOLD
Command24=ID_HISTO_CONTRAST
Command25=ID_HISTOGRAM_EQUAL
Command26=ID_MEANFILTER
Command27=ID_MEDIANFILTER
Command28=ID_GAUSSIAN_SMOOTHING
Command29=ID_CONSERVE_FILTER
Command30=ID_SHARPENING
Command31=ID_ROBERT_CROSS
Command32=ID_SOBEL
Command33=ID_ZERO_CROSSING
Command34=ID_CANNYEDGE_OPERATOR
Command35=ID_HOUGH_TRANSFORM
Command36=ID_ROTATION
Command37=ID_INTERPOLATION
Command38=ID_FLIP_X
Command39=ID_FLIP_Y
Command40=ID_FLIP_DIAGONAL
Command41=ID_AFFINE_TRANSFORM
Command42=ID_KMEAN_CLUSTERING
Command43=ID_KMEAN_CLUSTERING_COLOR
Command44=ID_CONNECTED_LABELING
Command45=ID_BINARY_DILATION
Command46=ID_GRAY_DILATION
Command47=ID_BINARY_EROSION
Command48=ID_GRAY_EROSION
Command49=ID_EM_OPERATION
Command50=ID_WINDOW_NEW
Command51=ID_WINDOW_CASCADE
Command52=ID_WINDOW_TILE_HORZ
Command53=ID_WINDOW_ARRANGE
Command54=ID_EDIT_UNDO
Command55=ID_EDIT_CUT
Command56=ID_EDIT_COPY
Command57=ID_EDIT_PASTE
Command58=ID_APP_ABOUT
Command59=ID_VIEW_TOOLBAR
Command60=ID_VIEW_STATUS_BAR
Command61=ID_MEANSHIFT
CommandCount=61

[ACL:IDR_MAINFRAME]
Type=1
Class=CMainFrame
Command1=ID_FILE_NEW
Command2=ID_FILE_OPEN
Command3=ID_FILE_SAVE
Command4=ID_FILE_PRINT
Command5=ID_EDIT_UNDO
Command6=ID_EDIT_CUT
Command7=ID_EDIT_COPY
Command8=ID_EDIT_PASTE
Command9=ID_EDIT_UNDO
Command10=ID_EDIT_CUT
Command11=ID_EDIT_COPY
Command12=ID_EDIT_PASTE
Command13=ID_NEXT_PANE
Command14=ID_PREV_PANE
CommandCount=14

[DLG:IDD_INPUT]
Type=1
Class=CInputDialog
ControlCount=5
Control1=IDOK,button,1342242817
Control2=IDCANCEL,button,1342242816
Control3=IDC_STATIC,static,1342308352
Control4=IDC_EDIT1,edit,1350631552
Control5=IDC_SPIN2,msctls_updown32,1342177463

[CLS:CInputDialog]
Type=0
HeaderFile=InputDialog.h
ImplementationFile=InputDialog.cpp
BaseClass=CDialog
Filter=D
LastObject=IDOK
VirtualFilter=dWC

[DLG:IDD_HISTOGRAM]
Type=1
Class=CHistogram
ControlCount=6
Control1=IDC_HISTO,button,1342242816
Control2=IDC_RADIO_RED,button,1342177289
Control3=IDC_RADIO_GREEN,button,1342177289
Control4=IDC_RADIO_BLUE,button,1342177289
Control5=IDC_RADIO_GRAY,button,1342177289
Control6=IDC_STATIC,button,1342177287

[CLS:CHistogram]
Type=0
HeaderFile=Histogram.h
ImplementationFile=Histogram.cpp
BaseClass=CDialog
Filter=D
LastObject=CHistogram
VirtualFilter=dWC

[CLS:CChapter1]
Type=0
HeaderFile=Chapter1.h
ImplementationFile=Chapter1.cpp
BaseClass=generic CWnd
Filter=W
LastObject=CChapter1

