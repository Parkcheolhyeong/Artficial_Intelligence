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
	ON_COMMAND(ID_CHAP3_6, &CImageProToolView::On4_2)
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

	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();	// 불러온 문서변수 생성
	ASSERT_VALID(pDoc);

	int i, j;

	int width = pDoc->m_Width;	// 불러온 문서의 너비추출
	int height = pDoc->m_Height;// 불러온 문서의 높이추출

	BYTE* pData = pDoc->m_pDib->GetBitsAddress();	// 문서의 화소주소값을 pData에 저장

	RGBptr** ptr1 = new RGBptr*[height];	//pData의 값을 복사할 **ptr1 생성
		/* **가 가르치는 *의 값 생성*/
	for (int i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
	}

	Seperate_RGB(pData, ptr1);	// pData의 값을 **ptr1에 저장

	int x, y;
	double scale = 3.5;
	int new_height = (int)(height * scale);
	int new_width = (int)(width * scale);

	RGBptr **ptr2 = new RGBptr*[new_height];
	for (i = 0; i < new_height; i++)
		ptr2[i] = new RGBptr[new_width];

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			x = (int)new_width*j / width;
			y = (int)new_height*i / height;
			ptr2[y][x] = ptr1[i][j];
		}

	if (scale < 1)
	{
		for (i = 0; i < height; i++)
			for (j = 0; j < width; j++) ptr1[i][j].r = ptr1[i][j].g = ptr1[i][j].b = 0;

		for (i = 0; i < new_height; i++)
			for (j = 0; j < new_width; j++)
				ptr1[i][j] = ptr2[i][j];

	}
	else {
		for (i = 0; i < height; i++)
			for (j = 0; j < width; j++)
				ptr1[i][j] = ptr2[i][j];
	}

	SetRGBptr(pData, ptr1, width, height);

	for (i = 0; i < height; i++)
	{
		delete[] ptr1[i];
	}

	for (i = 0; i < new_height; i++)
		delete[] ptr2[i];

	delete[] ptr1;
	delete[] ptr2;

	Invalidate();

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
	RGBptr** ptr2 = new RGBptr*[height];	//pData의 값을 복사할 **ptr1 생성

	BYTE** image = new BYTE*[height];	// ptr1의 값을 조작할 때 저장할 **image생성
	BYTE** temp_image = new BYTE*[height];
	BYTE** rValue = new BYTE*[height];

	/* **가 가르치는 *의 값 생성*/
	for (int i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		ptr2[i] = new RGBptr[width];
		image[i] = new BYTE[width];
		temp_image[i] = new BYTE[width];
		rValue[i] = new BYTE[width];
	}

	Seperate_RGB(pData, ptr1);	// pData의 값을 **ptr1에 저장

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			image[i][j] = (BYTE)Saturation((ptr1[i][j].r + ptr1[i][j].g + ptr1[i][j].b) / 3);

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			if (ptr1[i][j].r > 190)
				temp_image[i][j] = 255;
			else
				temp_image[i][j] = 0;

			ptr1[i][j].r = ptr1[i][j].g = ptr1[i][j].b = image[i][j];
		}
	//팽창->침식(열림)
	dilation(ptr1, temp_image, width, height);

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			ptr2[i][j] = ptr1[i][j];

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			if (ptr2[i][j].r > 190)
				temp_image[i][j] = 255;
			else
				temp_image[i][j] = 0;

			ptr2[i][j].r = ptr2[i][j].g = ptr2[i][j].b = image[i][j];
		}

	erosion(ptr2, temp_image, width, height);

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			image[i][j] = ptr1[i][j].r - ptr2[i][j].r;

			if (image[i][j] == 0) image[i][j] = 255;
			else if (image[i][j] == 255) image[i][j] = 0;

			ptr1[i][j].r = ptr1[i][j].g = ptr1[i][j].b = image[i][j];
		}

	SetRGBptr(pData, ptr1, width, height);

	for (i = 0; i < height; i++)
	{
		delete[] ptr1[i];
		delete[] image[i];
		delete[] temp_image[i];
	}

	delete[] ptr1;
	delete[] image;
	delete[] temp_image;

	Invalidate();
}


void CImageProToolView::dilation(RGBptr** ptr1, BYTE** temp_image, int width, int height)
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.

	BYTE** image = new BYTE*[height];	// ptr1의 값을 조작할 때 저장할 **image생성

	int i, j;
	/* **가 가르치는 *의 값 생성*/
	for (int i = 0; i < height; i++)
	{
		image[i] = new BYTE[width];
	}

	for (i = 1; i < height - 1; i++)
		for (j = 1; j < width - 1; j++)
		{
			if (temp_image[i - 1][j - 1] == 0 && temp_image[i - 1][j] == 0 &&
				temp_image[i - 1][j + 1] == 0 && temp_image[i][j - 1] == 0 &&
				temp_image[i][j + 1] == 0 && temp_image[i + 1][j - 1] == 0 &&
				temp_image[i + 1][j] == 0 && temp_image[i + 1][j + 1] == 0)
				image[i][j] = 0;
			else
				image[i][j] = 255;

		}

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			ptr1[i][j].r = image[i][j];
			ptr1[i][j].g = image[i][j];
			ptr1[i][j].b = image[i][j];

		}

	for (i = 0; i < height; i++)
	{
		delete[] image[i];

	}

	delete[] image;	
}


void CImageProToolView::erosion(RGBptr** ptr1, BYTE** temp_image, int width, int height)
{
	// TODO: 여기에 구현 코드 추가.
	BYTE** image = new BYTE*[height];	// ptr1의 값을 조작할 때 저장할 **image생성

	int i, j;
	/* **가 가르치는 *의 값 생성*/
	for (int i = 0; i < height; i++)
	{
		image[i] = new BYTE[width];
	}

	for (i = 1; i < height - 1; i++)
		for (j = 1; j < width - 1; j++)
		{
			if (temp_image[i - 1][j - 1] == 255 && temp_image[i - 1][j] == 255 &&
				temp_image[i - 1][j + 1] == 255 && temp_image[i][j - 1] == 255 &&
				temp_image[i][j + 1] == 255 && temp_image[i + 1][j - 1] == 255 &&
				temp_image[i + 1][j] == 255 && temp_image[i + 1][j + 1] == 255)
				image[i][j] = 255;
			else
				image[i][j] = 0;

		}

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			ptr1[i][j].r = image[i][j];
			ptr1[i][j].g = image[i][j];
			ptr1[i][j].b = image[i][j];

		}

	for (i = 0; i < height; i++)
	{
		delete[] image[i];

	}

	delete[] image;
}


void CImageProToolView::On4_2()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	/* 1.잡음제거(Median) 2. 전역 임계 3. 영상 축소 4. 모폴로지 2회 5. 영역 레이블링*/
	CImageProToolView::OnMedianfiltering();
	CImageProToolView::OnGlobalthresholding();
	CImageProToolView::OnNearestscaling();
	CImageProToolView::OnErosion();
	CImageProToolView::OnErosion();
	CImageProToolView::OnConnectedlabeling();
}


void CImageProToolView::OnMedianfiltering()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();
	ASSERT_VALID(pDoc);
	int i, j, m, n;

	int width = pDoc->m_Width;
	int height = pDoc->m_Height;


	BYTE* pData = pDoc->m_pDib->GetBitsAddress();
	RGBptr** ptr1 = new RGBptr*[height];
	int** image = new int*[height];
	int** out_image = new int*[height];

	for (i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		image[i] = new int[width];
		out_image[i] = new int[width];
	}

	Seperate_RGB(pData, ptr1);

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			image[i][j] = (BYTE)Saturation((ptr1[i][j].r + ptr1[i][j].g + ptr1[i][j].b) / 3);

	int median = 0;

	int NUM = 9;
	int* pixel = new int[NUM];
	int pidx;

	for (i = 1; i < height - 1; i++)
		for (j = 1; j < width - 1; j++)
		{
			pidx = 0;

			for (m = -1; m <= 1; m++)
				for (n = -1; n <= 1; n++)
					pixel[pidx++] = (int)image[i + m][j + n];

			int subi, subj, tmp;

			for (subi = 0; subi < NUM - 1; subi++)
				for (subj = subi + 1; subj < NUM; subj++)
					if (pixel[subi] > pixel[subj])
					{
						tmp = pixel[subi];
						pixel[subi] = pixel[subj];
						pixel[subj] = tmp;
					}

			if (NUM % 2 != 0)
				median = pixel[NUM / 2];
			else
				median = (pixel[NUM / 2 - 1] + pixel[NUM / 2]) / 2;

			out_image[i][j] = (unsigned char)Saturation(median);
		}

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			ptr1[i][j].r = ptr1[i][j].g = ptr1[i][j].b = out_image[i][j];

	SetRGBptr(pData, ptr1, width, height);

	for (i = 0; i < height; i++)
	{
		delete[] ptr1[i];
		delete[] image[i];
		delete[] out_image[i];
	}

	delete[] ptr1;
	delete[] out_image;
	delete[] image;
	Invalidate();
}


void CImageProToolView::OnGlobalthresholding()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();
	ASSERT_VALID(pDoc);
	int i, j, T1 = 150, T2 = 0, T0 = 1;

	int width = pDoc->m_Width;
	int height = pDoc->m_Height;


	BYTE* pData = pDoc->m_pDib->GetBitsAddress();
	RGBptr** ptr1 = new RGBptr*[height];

	for (i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
	}

	Seperate_RGB(pData, ptr1);


	int* Y = new int[height * width];
	int* Bi = new int[height * width];
	memset(Y, 0, height*width);
	memset(Bi, 0, height*width);

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			Y[i*width + j] = Saturation((ptr1[i][j].r + ptr1[i][j].g + ptr1[i][j].b) / 3);

			if (Y[i] > T1) Bi[i] = 255;
			else Bi[i] = 0;
		}

	while (1)
	{
		int T2 = Thresholding_Update(height, width, Y, Bi, T1);

		if (abs(T1 - T2) < T0)
		{
			for (i = 0; i < height; i++)
			{
				for (j = 0; j < width; j++)
				{
					if (Y[i*width + j] > T2) Bi[i*width + j] = 255;
					else Bi[i*width + j] = 0;
				}
			}
			break;
		}
		else
		{
			T1 = T2;

			for (i = 0; i < height; i++)
				for (j = 0; j < width; j++)
				{
					if (Y[i*width + j] > T2) Bi[i*width + j] = 255;
					else Bi[i*width + j] = 0;
				}
		}
	}

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			ptr1[i][j].r = Bi[i*width + j];
			ptr1[i][j].g = Bi[i*width + j];
			ptr1[i][j].b = Bi[i*width + j];
		}

	SetRGBptr(pData, ptr1, width, height);

	for (i = 0; i < height; i++)
		delete[] ptr1[i];

	delete[] ptr1;
	delete[] Y;
	delete[] Bi;
	Invalidate();
}
int CImageProToolView::Thresholding_Update(int height, int width, int* Y, int *B, int T)
{
	int i, count1 = 0, count2 = 0, sum1 = 0, sum2 = 0;

	for (i = 0; i < height * width; i++)
	{
		if (B[i] == 255) { sum1 += Y[i]; count1++; }
		else { sum2 += Y[i]; count2++; }
	}

	int ave1 = sum1 / count1;
	int ave2 = sum2 / count2;

	int T2 = (ave1 + ave2) / 2;

	return T2;
}


void CImageProToolView::OnErosion()
{
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();	// 불러온 문서변수 생성
	ASSERT_VALID(pDoc);

	int i, j;

	int width = pDoc->m_Width;	// 불러온 문서의 너비추출
	int height = pDoc->m_Height;// 불러온 문서의 높이추출

	BYTE* pData = pDoc->m_pDib->GetBitsAddress();	// 문서의 화소주소값을 pData에 저장

	RGBptr** ptr1 = new RGBptr*[height];	//pData의 값을 복사할 **ptr1 생성
	BYTE** image = new BYTE*[height];	// ptr1의 값을 조작할 때 저장할 **image생성
	BYTE** temp_image = new BYTE*[height];

	/* **가 가르치는 *의 값 생성*/
	for (int i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		image[i] = new BYTE[width];
		temp_image[i] = new BYTE[width];
	}

	Seperate_RGB(pData, ptr1);	// pData의 값을 **ptr1에 저장

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			image[i][j] = (BYTE)Saturation((ptr1[i][j].r + ptr1[i][j].g + ptr1[i][j].b) / 3);

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			if (image[i][j] > 170)
				temp_image[i][j] = 255;
			else
				temp_image[i][j] = 0;
		}

	for (i = 1; i < height - 1; i++)
		for (j = 1; j < width - 1; j++)
		{
			if (temp_image[i - 1][j - 1] == 0 && temp_image[i - 1][j] == 0 &&
				temp_image[i - 1][j + 1] == 0 && temp_image[i][j - 1] == 0 &&
				temp_image[i][j + 1] == 0 && temp_image[i + 1][j - 1] == 0 &&
				temp_image[i + 1][j] == 0 && temp_image[i + 1][j + 1] == 0)
				image[i][j] = 0;
			else
				image[i][j] = 255;

		}

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			ptr1[i][j].r = image[i][j];
			ptr1[i][j].g = image[i][j];
			ptr1[i][j].b = image[i][j];

		}
	SetRGBptr(pData, ptr1, width, height);

	for (i = 0; i < height; i++)
	{
		delete[] ptr1[i];
		delete[] image[i];
		delete[] temp_image[i];

	}

	delete[] ptr1;
	delete[] image;
	delete[] temp_image;

	Invalidate();
}


void CImageProToolView::OnNearestscaling()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.


	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();	// 불러온 문서변수 생성
	ASSERT_VALID(pDoc);

	int i, j;

	int width = pDoc->m_Width;	// 불러온 문서의 너비추출
	int height = pDoc->m_Height;// 불러온 문서의 높이추출

	BYTE* pData = pDoc->m_pDib->GetBitsAddress();	// 문서의 화소주소값을 pData에 저장

	RGBptr** ptr1 = new RGBptr*[height];	//pData의 값을 복사할 **ptr1 생성
		/* **가 가르치는 *의 값 생성*/
	for (int i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
	}

	Seperate_RGB(pData, ptr1);	// pData의 값을 **ptr1에 저장

	
	int x, y;
	double scale = 0.5;
	int new_height = (int)(height * scale);
	int new_width = (int)(width * scale);

	RGBptr **ptr2 = new RGBptr*[new_height];
	for (i = 0; i < new_height; i++)
		ptr2[i] = new RGBptr[new_width];

	for (i = 0; i < new_height; i++)
		for (j = 0; j < new_width; j++)
		{
			x = (int)width*j / new_width;
			y = (int)height*i / new_height;
			ptr2[i][j] = ptr1[y][x];
		}

	if (scale < 1)
	{
		for (i = 0; i < height; i++)
			for (j = 0; j < width; j++) ptr1[i][j].r = ptr1[i][j].g = ptr1[i][j].b = 0;

		for (i = 0; i < new_height; i++)
			for (j = 0; j < new_width; j++)
				ptr1[i][j] = ptr2[i][j];

	}
	else {
		for (i = 0; i < height; i++)
			for (j = 0; j < width; j++)
				ptr1[i][j] = ptr2[i][j];
	}

	SetRGBptr(pData, ptr1, width, height);

	for (i = 0; i < height; i++)
	{
		delete[] ptr1[i];
	}

	for (i = 0; i < new_height; i++)
		delete[] ptr2[i];

	delete[] ptr1;
	delete[] ptr2;

	Invalidate();
}


void CImageProToolView::OnConnectedlabeling()
{
	// TODO: 여기에 구현 코드 추가.
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();
	ASSERT_VALID(pDoc);

	int width = pDoc->m_Width;
	int height = pDoc->m_Height;

	BYTE* pData = pDoc->m_pDib->GetBitsAddress();
	RGBptr** ptr1 = new RGBptr*[height];
	BYTE** image = new BYTE*[height];
	int i, j;

	for (i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		image[i] = new BYTE[width];
	}

	Seperate_RGB(pData, ptr1);

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			image[i][j] = (BYTE)Saturation((ptr1[i][j].r + ptr1[i][j].g + ptr1[i][j].b) / 3);
			if (image[i][j] < 170) image[i][j] = 255;
			else image[i][j] = 0;
		}
	//minimum number of pixels in region that to be removed 
	int minRegionCount = 10;

	int** Label = new int*[height];
	for (i = 0; i < height; i++)
	{
		Label[i] = new int[width];
		memset(Label[i], 0, sizeof(int)*width);
	}

	int num, left, top, k;
	int *r, *area;
	r = new int[height*width];
	area = new int[width*height];
	memset(r, 0, sizeof(int)*height*width);
	memset(area, 0, sizeof(int)*height*width);

	//threholding to make binary image
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			if (image[i][j] > 180) { Label[i][j] = 1; }
			else { Label[i][j] = -1; }

	//do not label of boundary of image
	for (j = 0; j < width; j++) {
		Label[0][j] = -1;
		Label[height - 1][j] = -1;
	}
	for (i = 0; i < height; i++) {
		Label[i][0] = -1;
		Label[i][width - 1] = -1;
	}

	num = 0; //initial lable number
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			if (i > 0 && j > 0) {
				// scan and if the pixel value is over 0
				if (Label[i][j] >= 1) {
					left = Label[i][j - 1]; //Examine Left and 
					top = Label[i - 1][j];  //Examine top lable
					if (left == -1 && top != -1) {  //if only top label
							//lable is assigned as the top lable
						Label[i][j] = r[top];
					}
					//if only left label
					else if (left != -1 && top == -1) {
						//lable is assigned as the left lable
						Label[i][j] = r[left];
					}
					//if no lable left and top
					else if (left == -1 && top == -1) {
						num++;             //increase label number +1
							   //save label number to equivalent table
						r[num] = num;
						//assign the label value to label image       
						Label[i][j] = r[num];
					}
					//In case left and top have values sametime  
					//1) if labels are same, assign as the left value
					//2) otherwise assign as the smaller value
					// 변경
					else if (left != -1 && top != -1) {
						if (r[left] == r[top]) {
							Label[i][j] = r[left];
						}
						else if (r[left] > r[top]) {
							Label[i][j] = r[top];
							// also lable is changed
							r[left] = r[top];
						}
						else {
							Label[i][j] = r[left];
							r[top] = r[left];
						}
					}
				}
			}
		}
	}
	for (k = 1; k <= num; k++) { //label is re-arranged
		   //if k and its label number is not same, new label is assigned 
		if (k != r[k]) r[k] = r[r[k]];
		area[k] = 0;
	}

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++) {
			if (Label[i][j] > 0) {
				//new label image is assigned as new labels 
				Label[i][j] = r[Label[i][j]];
				// sum of all numbers of each label ==> to remove small label regions
				area[Label[i][j]]++;
			}
		}
	int cnt = 0;
	for (k = 1; k <= num; k++) {
		//if the number of region's label is below threshold
		//remove it and re-assign the label number
		if (area[k] >= minRegionCount)
		{
			r[k] = cnt++;
		}
		else r[k] = -1;
	}
	cnt--;

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++) {
			if (Label[i][j] > 0)
				//re-assign the label number
				Label[i][j] = r[Label[i][j]];
		}
	//화면 출력을 위한 과정
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			//if (Label[i][j] == 12){
			ptr1[i][j].r = Label[i][j] * 10;
			ptr1[i][j].g = Label[i][j] * 10;
			ptr1[i][j].b = Label[i][j] * 10;
			//}
		}

	SetRGBptr(pData, ptr1, width, height);

	delete[] r;
	delete[] area;

	for (i = 0; i < height; i++)
	{
		delete[] Label[i];
		delete[] image[i];
		delete[] ptr1[i];
	}
	delete[] image;
	delete[] Label;
	delete[] ptr1;

	//print label number
	CString str;
	str.Format(_T("레이블 개수 = %d"), cnt);
	AfxMessageBox(str);
	Invalidate();
}




void CImageProToolView::OnUpdate()
{
	// TODO: 여기에 구현 코드 추가.
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();
	ASSERT_VALID(pDoc);

	int width = pDoc->m_Width;
	int height = pDoc->m_Height;

	RGBptr**ptr1 = new RGBptr*[height];
	BYTE* pData = pDoc->m_pDib->GetBitsAddress();

	for (int i = 0; i < height; i++)
		ptr1[i] = new RGBptr[width];

	Seperate_RGB(pData, ptr1);
	SetRGBptr(pData, ptr1, width, height);
}
