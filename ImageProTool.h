// ImageProTool.h : main header file for the IMAGEPROTOOL application
//

#if !defined(AFX_IMAGEPROTOOL_H__23AF0E14_BA72_11D4_B429_00001C049D51__INCLUDED_)
#define AFX_IMAGEPROTOOL_H__23AF0E14_BA72_11D4_B429_00001C049D51__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#ifndef __AFXWIN_H__
	#error include 'stdafx.h' before including this file for PCH
#endif

#include "resource.h"       // main symbols

/////////////////////////////////////////////////////////////////////////////
// CImageProToolApp:
// See ImageProTool.cpp for the implementation of this class
//
class CDIB;
class CImageProToolApp : public CWinApp
{
public:
    CDIB* m_pNewDib;
	CImageProToolApp();

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CImageProToolApp)
	public:
	virtual BOOL InitInstance();
	//}}AFX_VIRTUAL

// Implementation
	//{{AFX_MSG(CImageProToolApp)
	afx_msg void OnAppAbout();
		// NOTE - the ClassWizard will add and remove member functions here.
		//    DO NOT EDIT what you see in these blocks of generated code !
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

// 전역 함수 선언

void AfxNewImage(CDIB* dib);
/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_IMAGEPROTOOL_H__23AF0E14_BA72_11D4_B429_00001C049D51__INCLUDED_)
