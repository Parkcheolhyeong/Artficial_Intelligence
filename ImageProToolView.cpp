// ImageProToolView.cpp : implementation of the CImageProToolView class
// Git master
// GIt hotfix

/*헤더파일 호출*/
#include "stdafx.h"
#include "ImageProTool.h"
#include "Dib.h"
#include "math.h"
#include "time.h"
#include "ImageProToolDoc.h"
#include "ImageProToolView.h"
#include "Histogram.h"

/* 수학적 수식 및 고정값 선언 */
const double PI = acos(-1.0);
const int N=15;


#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif



//========================================
/////////////////////////////////////////////////////////////////////////////
// CImageProToolView

/*DIalog의 이벤트처리기 설정*/
IMPLEMENT_DYNCREATE(CImageProToolView, CScrollView)

BEGIN_MESSAGE_MAP(CImageProToolView, CScrollView)
	//{{AFX_MSG_MAP(CImageProToolView)
	ON_COMMAND(ID_FILE_OPEN, OnFileOpen)

	//}}AFX_MSG_MAP
	// Standard printing commands
	ON_COMMAND(ID_FILE_PRINT, CScrollView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, CScrollView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, CScrollView::OnFilePrintPreview)
	ON_COMMAND(ID_CHAP_HISTOGRAM32884, &CImageProToolView::OnHistogram)
	ON_COMMAND(ID_CHAP3_4, &CImageProToolView::On3_3)
	ON_COMMAND(ID_CHAP3_3, &CImageProToolView::On3_2)
	ON_COMMAND(ID_CHAP3_5, &CImageProToolView::On4_1)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CImageProToolView construction/destruction

CImageProToolView::CImageProToolView()
{
  Is_FileOpen = false;
}

CImageProToolView::~CImageProToolView()
{
}

BOOL CImageProToolView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	return CScrollView::PreCreateWindow(cs);
}

/////////////////////////////////////////////////////////////////////////////
// CImageProToolView drawing

void CImageProToolView::OnDraw(CDC* pDC)
{
	CImageProToolDoc* pDoc = GetDocument();	// 파일로드
  	ASSERT_VALID(pDoc);	// 객체 유효성 검사

	if(Is_FileOpen)	// 파일이 열렸을 때 작동
	{
  
		CSize sizeTotal;	// 너비, 높이 값 저장을 위한 CSize 변수 생성
		
		sizeTotal.cx =pDoc->m_Width; // 파일의 너비저장
		sizeTotal.cy =pDoc->m_Height;// 파일의 높이저장
		SetScrollSizes(MM_TEXT, sizeTotal);// 스크롤 사이즈 설정
    
		GetParentFrame()->RecalcLayout();//부모윈도우으로 하여금 레이아웃 재설정 요청
		ResizeParentToFit(FALSE);		 //프레임을 불러오는 파일에 맞게 설정
   	
		CPalette* pOldPalette = pDC->SelectPalette(pDoc->m_pDib->m_pPalDib, FALSE);	// 팔레트 설정
		pDC->RealizePalette();
		pDoc->m_pDib->Draw(pDC, 0, 0,pDoc->m_Width, pDoc->m_Height);	// 그리기 가능한 범위지정

	}
}


void CImageProToolView::OnInitialUpdate()
{
	CScrollView::OnInitialUpdate();	// 부모 클래스의 OnInitialUpdate호출 
    CImageProToolDoc* pDoc = GetDocument();
	CSize sizeTotal;
	
	/*파일의 너비, 높이가 없다면 100,100으로 있다면 불러온 파일의 너비, 높이로 설정*/
	if(pDoc->m_Height<=0 || pDoc->m_Width<=0)
	{sizeTotal.cx =100;
	sizeTotal.cy =100;}
	else {
    sizeTotal.cx =pDoc->m_Height;
	sizeTotal.cy =pDoc->m_Width;
	}
	SetScrollSizes(MM_TEXT, sizeTotal);
    
	GetParentFrame()->RecalcLayout();
	ResizeParentToFit(FALSE);
}

/////////////////////////////////////////////////////////////////////////////
// CImageProToolView printing

BOOL CImageProToolView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// default preparation
	return DoPreparePrinting(pInfo);
}

void CImageProToolView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add extra initialization before printing
}

void CImageProToolView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add cleanup after printing
}

/////////////////////////////////////////////////////////////////////////////
// CImageProToolView diagnostics

#ifdef _DEBUG
void CImageProToolView::AssertValid() const
{
	CScrollView::AssertValid();
}

void CImageProToolView::Dump(CDumpContext& dc) const
{
	CScrollView::Dump(dc);
}

CImageProToolDoc* CImageProToolView::GetDocument() // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CImageProToolDoc)));
	return (CImageProToolDoc*)m_pDocument;
}
#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CImageProToolView message handlers

/*파일을 열때 유효검사 및 bool값 갱신*/
void CImageProToolView::OnFileOpen() 
{   

	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();
	ASSERT_VALID(pDoc);

	pDoc->OnFileOpen();
	Is_FileOpen = true;
	Invalidate();
}


//-----------------------------------------------------------------------------------------------Seperate and Set RGB

void CImageProToolView::Seperate_RGB(BYTE* Data, RGBptr** ptr)
{
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();	// 파일로드
	ASSERT_VALID(pDoc);	// 객체의 유효성 검사
	int width = pDoc->m_Width;	// 파일의 너비저장
	int height = pDoc->m_Height;// 파일의 높이저장

	int i,j,y,x;	// for문을 위한 변수선언

	/* pointer인 Data 변수에서 double pointer ptr로 값저장(r,g,b로 분리된 구조체 멤버변수에 저장)  */
	for(i=height-1,y=0;i>=0;i--,y++)
		for(j=0,x=0;j<width;j++,x++)
		{  
			ptr[y][x].b=Data[i*width*3+j*3];
			ptr[y][x].g=Data[i*width*3+j*3+1];
			ptr[y][x].r=Data[i*width*3+j*3+2];
		}
}
void CImageProToolView::SetRGBptr(BYTE* pData, RGBptr** ptr1,int width, int height)
{
	int i,j,y,x;	// for문을 위한 변수선언
	/* double pointer ptr1 값을 pData에 저장 */
	for(i=height-1,y=0;i>=0;i--,y++)
		for(j=0,x=0;j<width;j++,x++)
		{
			pData[i*width*3+j*3]=(BYTE)(ptr1[y][x].b);
			pData[i*width*3+j*3+1]=(BYTE)(ptr1[y][x].g);
			pData[i*width*3+j*3+2]=(BYTE)(ptr1[y][x].r);

		}	
}


//-------------------------------------------------------------------------------------------------------------------


void CImageProToolView::OnHistogram()
{
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();
	ASSERT_VALID(pDoc);

	int width = pDoc->m_Width;
	int height = pDoc->m_Height;
	int i, j;

	BYTE* pData = pDoc->m_pDib->GetBitsAddress();
	RGBptr** ptr1 = new RGBptr*[height];

	for (i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
	}

	Seperate_RGB(pData, ptr1);

	int tempR[256], tempG[256], tempB[256], tempgray[256];
	float histoR[256] = { 0.f, }, histoG[256] = { 0.f, }, histoB[256] = { 0.f, }, histogray[256] = { 0.f, };
	memset(tempgray, 0, sizeof(int) * 256);
	memset(tempR, 0, sizeof(int) * 256);
	memset(tempG, 0, sizeof(int) * 256);
	memset(tempB, 0, sizeof(int) * 256);
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			int value = Saturation((ptr1[i][j].r + ptr1[i][j].g + ptr1[i][j].b) / 3);
			tempgray[value]++;
			tempR[ptr1[i][j].r]++;
			tempG[ptr1[i][j].g]++;
			tempB[ptr1[i][j].b]++;
		}

	// 히스토그램 정규화(histogram normalization)

	float area = (float)width*height;
	for (i = 0; i < 256; i++)
	{
		histogray[i] = tempgray[i] / (float)area;
		histoR[i] = (float)tempR[i] / (float)area;
		histoG[i] = (float)tempG[i] / (float)area;
		histoB[i] = (float)tempB[i] / (float)area;
	}



	Histogram dlg; // 객체변수
	dlg.SetImage(histogray, histoR, histoG, histoB);
	dlg.DoModal();
}

void CImageProToolView::On3_2()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.

}

void CImageProToolView::On3_3()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();
	ASSERT_VALID(pDoc);

	int i, j, m, n;

	int width = pDoc->m_Width;
	int height = pDoc->m_Height;

	BYTE* pData = pDoc->m_pDib->GetBitsAddress();

	RGBptr** ptr1 = new RGBptr*[height];
	RGBptr** ptr2 = new RGBptr*[height];

	for (i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		ptr2[i] = new RGBptr[width];
	}

	Seperate_RGB(pData, ptr1);

	int center_x = width / 2;
	int center_y = height / 2;

	double angle = 3.14159265 / 180 * 45;

	double x1, y1, x2, y2, a, b;

	int x, y;

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			x1 = (j - center_x)*cos(angle);
			y1 = -1 * ((i - center_y)*sin(angle));

			x = (int)(x1 + y1 + center_x);

			x2 = (j - center_x)*sin(angle);
			y2 = (i - center_y)*cos(angle);
				
			y = (int)(x2 + y2 + center_y);

			if (y < 0 || y >= (height - 1) || x < 0 || x >= (width - 1))
			{
				ptr2[i][j].r = ptr2[i][j].g = ptr2[i][j].b = 0;
			}
			else
			{
				ptr2[i][j].r = (double)ptr1[y][x].r;
				ptr2[i][j].g = (double)ptr1[y][x].g;
				ptr2[i][j].b = (double)ptr1[y][x].b;
			}
		}
	// scale


//	Seperate_RGB(pData, ptr1);	// pData의 값을 **ptr1에 저장

	double scale = 0.3;
	int new_height = (int)(height * scale);
	int new_width = (int)(width * scale);

	*ptr1 = new RGBptr[new_height];
	for (i = 0; i < new_height; i++)
		ptr1[i] = new RGBptr[new_width];

	for (i = 0; i < new_height; i++)
		for (j = 0; j < new_width; j++)
		{
			x = (int)width*j / new_width;
			y = (int)height*i / new_height;
			ptr1[i][j] = ptr2[y][x];
		}

	if (scale < 1)
	{
		for (i = 0; i < height; i++)
			for (j = 0; j < width; j++) ptr2[i][j].r = ptr2[i][j].g = ptr2[i][j].b = 0;

		for (i = 0; i < new_height; i++)
			for (j = 0; j < new_width; j++)
				ptr2[i][j] = ptr1[i][j];

	}
	else {
		for (i = 0; i < height; i++)
			for (j = 0; j < width; j++)
				ptr2[i][j] = ptr1[i][j];
	}

	SetRGBptr(pData, ptr2, width, height);

	for (i = 0; i < height; i++)
	{
		delete[] ptr2[i];
	}

	for (i = 0; i < new_height; i++)
		delete[] ptr1[i];

	delete[] ptr1;
	delete[] ptr2;

	Invalidate();
}



void CImageProToolView::On4_1()
{
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();	// 불러온 문서변수 생성
	ASSERT_VALID(pDoc);

	int i, j;

	int width = pDoc->m_Width;	// 불러온 문서의 너비추출
	int height = pDoc->m_Height;// 불러온 문서의 높이추출

	BYTE* pData = pDoc->m_pDib->GetBitsAddress();	// 문서의 화소주소값을 pData에 저장

	RGBptr** ptr1 = new RGBptr*[height];	//pData의 값을 복사할 **ptr1 생성
	BYTE** dilation_image = new BYTE*[height];	// ptr1의 값을 조작할 때 저장할 **image생성
	BYTE** binary_image = new BYTE*[height];
	BYTE** erosion_image = new BYTE*[height];

	/* **가 가르치는 *의 값 생성*/
	for (int i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		dilation_image[i] = new BYTE[width];
		erosion_image[i] = new BYTE[width];
		binary_image[i] = new BYTE[width];
	}

	Seperate_RGB(pData, ptr1);	// pData의 값을 **ptr1에 저장

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			dilation_image[i][j] = (BYTE)Saturation(ptr1[i][j].r);

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			if (dilation_image[i][j] > 170)
				binary_image[i][j] = 255;
			else
				binary_image[i][j] = 0;
		}

	for (i = 1; i < height - 1; i++)
		for (j = 1; j < width - 1; j++)
		{
			if (binary_image[i - 1][j - 1] == 0 && binary_image[i - 1][j] == 0 &&
				binary_image[i - 1][j + 1] == 0 && binary_image[i][j - 1] == 0 &&
				binary_image[i][j + 1] == 0 && binary_image[i + 1][j - 1] == 0 &&
				binary_image[i + 1][j] == 0 && binary_image[i + 1][j + 1] == 0)
				dilation_image[i][j] = 0;
			else
				dilation_image[i][j] = 255;

			if (binary_image[i - 1][j - 1] == 255 && binary_image[i - 1][j] == 255 &&
				binary_image[i - 1][j + 1] == 255 && binary_image[i][j - 1] == 255 &&
				binary_image[i][j + 1] == 255 && binary_image[i + 1][j - 1] == 255 &&
				binary_image[i + 1][j] == 255 && binary_image[i + 1][j + 1] == 255)
				erosion_image[i][j] = 255;
			else
				erosion_image[i][j] = 0;
		}

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			ptr1[i][j].r = dilation_image[i][j] - erosion_image[i][j];
			ptr1[i][j].g = dilation_image[i][j] - erosion_image[i][j];
			ptr1[i][j].b = dilation_image[i][j] - erosion_image[i][j];

		}
	SetRGBptr(pData, ptr1, width, height);

	for (i = 0; i < height; i++)
	{
		delete[] ptr1[i];
		delete[] dilation_image[i];
		delete[] binary_image[i];

	}

	delete[] ptr1;
	delete[] dilation_image;
	delete[] binary_image;

	Invalidate();
}
