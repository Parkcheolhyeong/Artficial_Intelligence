// ImageProToolView.h : interface of the CImageProToolView class
//
/////////////////////////////////////////////////////////////////////////////

#if !defined(AFX_IMAGEPROTOOLVIEW_H__23AF0E1E_BA72_11D4_B429_00001C049D51__INCLUDED_)
#define AFX_IMAGEPROTOOLVIEW_H__23AF0E1E_BA72_11D4_B429_00001C049D51__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
class CDIB;


class CImageProToolView : public CScrollView
{
protected: // create from serialization only
	CImageProToolView();
	DECLARE_DYNCREATE(CImageProToolView)

// Attributes
public:
	CImageProToolDoc* GetDocument();

	struct RGBptr{
	 BYTE r;
	 BYTE g;
	 BYTE b;
	 }; 

	BOOL Is_FileOpen;
  
	void Seperate_RGB(BYTE* , RGBptr** );
	void SetRGBptr(BYTE* , RGBptr** ,int , int);
	template<typename T> 
	inline void swap(T& lha, T& rha)
	{
	T tmp = lha;
	lha = rha;
	rha = tmp;
	}



// Operations
public:
   
// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CImageProToolView)
	public:
	virtual void OnDraw(CDC* pDC);  // overridden to draw this view
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
	protected:
	virtual void OnInitialUpdate(); // called first time after construct
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);
	//}}AFX_VIRTUAL

// Implementation
public:
	virtual ~CImageProToolView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	//{{AFX_MSG(CImageProToolView)
	afx_msg void OnFileOpen();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnHistogram();

	afx_msg void OnTest();
	afx_msg void OnTrain();
	void trainSet(int k);
	afx_msg void OnTrainmode0();
	afx_msg void OnTrainmode1();
	afx_msg void OnTrainmode2();
	afx_msg void OnTrainmode4();
	afx_msg void OnTrainmode5();
	afx_msg void OnTrainmode6();
	afx_msg void OnTrainmode7();
	afx_msg void OnTrainmode8();
	afx_msg void OnTrainmode9();
	afx_msg void OnTrainmode3();
};

#ifndef _DEBUG  // debug version in ImageProToolView.cpp
inline CImageProToolDoc* CImageProToolView::GetDocument()
   { return (CImageProToolDoc*)m_pDocument; }
#endif

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_IMAGEPROTOOLVIEW_H__23AF0E1E_BA72_11D4_B429_00001C049D51__INCLUDED_)

/*************************************************************************
* 
* 이름 : Saturation(const T& value)
* 인자 : value - 정수 혹은 실수.
* 반환 : 0~255 사이의 값.
* 설명 : value 값이 0보다 작으면 0으로, 255보다 크면 255로 변경한다.
* 
*************************************************************************/

template<typename T>
inline T Saturation(const T& value)
{
	return ( (value > 255) ? 255 : ((value < 0) ? 0 : value) );
}

template<typename T>
inline T Wrapping(const T& value)
{
	return ( (value > 255) ? value-(255+1) : ((value < 0) ? 0 : value) );
}

