// ImageProToolDoc.cpp : implementation of the CImageProToolDoc class
//

#include "stdafx.h"
#include "ImageProTool.h"
#include "Dib.h"
#include "ImageProToolDoc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CImageProToolDoc

IMPLEMENT_DYNCREATE(CImageProToolDoc, CDocument)

BEGIN_MESSAGE_MAP(CImageProToolDoc, CDocument)
	//{{AFX_MSG_MAP(CImageProToolDoc)
	ON_COMMAND(ID_FILE_OPEN, OnFileOpen)
	//}}AFX_MSG_MAP
	
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CImageProToolDoc construction/destruction

CImageProToolDoc::CImageProToolDoc()
{
    m_pDib =NULL;

}

CImageProToolDoc::~CImageProToolDoc()
{
}

BOOL CImageProToolDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: add reinitialization code here
	// (SDI documents will reuse this document)

	return TRUE;
}



/////////////////////////////////////////////////////////////////////////////
// CImageProToolDoc serialization

void CImageProToolDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: add storing code here
	}
	else
	{
		// TODO: add loading code here
	}
}

/////////////////////////////////////////////////////////////////////////////
// CImageProToolDoc diagnostics

#ifdef _DEBUG
void CImageProToolDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CImageProToolDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CImageProToolDoc commands


void CImageProToolDoc::OnFileOpen()
{
	CFileDialog fileDlg(TRUE, "BMP", 0, NULL, 
                      "Bitmap Files(*.bmp)|*.bmp|"
                      "All Files(*.*)|*.*||", 
                      AfxGetMainWnd()); 
    if(fileDlg.DoModal() != IDOK)
    return;

  
		CString	szFileName;
	    //CString	szFileName = CString();
		CString Filename;
        		
       CFile* fp = new CFile() ;
		Filename = fileDlg.GetPathName();
		
		fp->Open(Filename,CFile::modeRead|CFile::typeBinary);

	    if(m_pDib !=NULL)
		      delete m_pDib;
		m_pDib = new CDIB();
      
		if(!m_pDib->Load(fp))
		{
			AfxMessageBox("Dib화일을 읽어 올수 없습니다.");
			return;
		}
	// 영상의 가로 세로 얻어온다.
        m_Width = m_pDib->DibWidth();
		m_Height= m_pDib->DibHeight();
		
	//-----------------------------------------------------	
	    fp->Close();	
}
