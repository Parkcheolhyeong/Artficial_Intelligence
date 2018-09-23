// ImageProToolDoc.h : interface of the CImageProToolDoc class
//
/////////////////////////////////////////////////////////////////////////////

#if !defined(AFX_IMAGEPROTOOLDOC_H__23AF0E1C_BA72_11D4_B429_00001C049D51__INCLUDED_)
#define AFX_IMAGEPROTOOLDOC_H__23AF0E1C_BA72_11D4_B429_00001C049D51__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
class CDIB;

class CImageProToolDoc : public CDocument
{
protected: // create from serialization only
	CImageProToolDoc();
	DECLARE_DYNCREATE(CImageProToolDoc)

// Attributes
public:
  CDIB*	m_pDib;
  int m_Width, m_Height;
// Operations
public:

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CImageProToolDoc)
	public:
	virtual BOOL OnNewDocument();
	virtual void Serialize(CArchive& ar);
	//}}AFX_VIRTUAL

// Implementation
public:
	virtual ~CImageProToolDoc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
public:
	//{{AFX_MSG(CImageProToolDoc)
	afx_msg void OnFileOpen();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
	
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_IMAGEPROTOOLDOC_H__23AF0E1C_BA72_11D4_B429_00001C049D51__INCLUDED_)
