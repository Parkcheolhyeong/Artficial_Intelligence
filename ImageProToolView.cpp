// ImageProToolView.cpp : implementation of the CImageProToolView class
// Git master
// GIt hotfix
#include "stdafx.h"
#include "ImageProTool.h"
#include "Dib.h"
#include "math.h"
#include "time.h"
#include "ImageProToolDoc.h"
#include "ImageProToolView.h"
#include "Histogram.h"


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

IMPLEMENT_DYNCREATE(CImageProToolView, CScrollView)

BEGIN_MESSAGE_MAP(CImageProToolView, CScrollView)
	//{{AFX_MSG_MAP(CImageProToolView)
	ON_COMMAND(ID_FILE_OPEN, OnFileOpen)

	//}}AFX_MSG_MAP
	// Standard printing commands
	ON_COMMAND(ID_FILE_PRINT, CScrollView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, CScrollView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, CScrollView::OnFilePrintPreview)
	ON_COMMAND(ID_CHAP_MEANFILTERING, &CImageProToolView::OnMeanfiltering)
	ON_COMMAND(ID_CHAP_MEDIANFILTERING, &CImageProToolView::OnMedianfiltering)
	ON_COMMAND(ID_ASSIGNMENT_TWO, &CImageProToolView::OnTwoAssignmentTwo)
	ON_COMMAND(ID_ASSIGNMENT_ONE, &CImageProToolView::OnTwoAssignmentOneStretching)
	ON_COMMAND(ID_ASSIGNMENT_THREE, &CImageProToolView::OnTwoAssignmentThree)
	ON_COMMAND(ID_CHAPTWOASSIGNMENT_TWOEQAUL, &CImageProToolView::OnTwoassignmentOneEqaul)
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
	CImageProToolDoc* pDoc = GetDocument();
  	ASSERT_VALID(pDoc);

	if(Is_FileOpen)
	{
  
    CSize sizeTotal;
		
	sizeTotal.cx =pDoc->m_Width;
	sizeTotal.cy =pDoc->m_Height;
	SetScrollSizes(MM_TEXT, sizeTotal);
    
	GetParentFrame()->RecalcLayout();
	ResizeParentToFit(FALSE);
   	
	CPalette* pOldPalette = pDC->SelectPalette(pDoc->m_pDib->m_pPalDib, FALSE);
	pDC->RealizePalette();
	pDoc->m_pDib->Draw(pDC, 0, 0,pDoc->m_Width, pDoc->m_Height);	

	}
}


void CImageProToolView::OnInitialUpdate()
{
	CScrollView::OnInitialUpdate();
    CImageProToolDoc* pDoc = GetDocument();
	CSize sizeTotal;
	
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
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();
	ASSERT_VALID(pDoc);
	int width = pDoc->m_Width;    
	int height = pDoc->m_Height; 

	int i,j,y,x;

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
	int i,j,y,x;
	for(i=height-1,y=0;i>=0;i--,y++)
		for(j=0,x=0;j<width;j++,x++)
		{
			pData[i*width*3+j*3]=(BYTE)(ptr1[y][x].b);
			pData[i*width*3+j*3+1]=(BYTE)(ptr1[y][x].g);
			pData[i*width*3+j*3+2]=(BYTE)(ptr1[y][x].r);

		}	
}


//-------------------------------------------------------------------------------------------------------------------


void CImageProToolView::OnMeanfiltering()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();	// 파일로드
	ASSERT_VALID(pDoc);	//객체의 유효성 검사
	int i, j;	// for문을 위한 정수 변수 생성

	int width = pDoc->m_Width;	// 파일의 너비저장 
	int height = pDoc->m_Height;// 파일의 높이저장


	BYTE* pData = pDoc->m_pDib->GetBitsAddress();// 파일의 픽셀을 가르치는 pointer 변수 생성
	RGBptr** ptr1 = new RGBptr*[height];	   	 // 파일의 각 화소값을 저장할 double pointer 변수 생성
	int** image = new int*[height];				 // **ptr1의 조작결과를 저장시킬 double pointer 변수 생성
	int** out_image = new int*[height];			 // **image의 조작결과를 저장시킬 double pointer 변수 생성

	/*각 double pointer 변수 별로 pointer 선언*/	
	for (i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		image[i] = new int[width];
		out_image[i] = new int[width];
	}

	Seperate_RGB(pData, ptr1);	//채도

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			image[i][j] = (BYTE)Saturation((ptr1[i][j].r + ptr1[i][j].g + ptr1[i][j].b) / 3);

	float mean = 0;

	float mean3x3[3][3] = { 1.0f / 9.0f, 1.0f / 9.0f , 1.0f / 9.0f,
						   1.0f / 9.0f , 1.0f / 9.0f  ,1.0f / 9.0f,
						   1.0f / 9.0f, 1.0f / 9.0f , 1.0f / 9.0f };
	
	int margin;

	margin = sqrt(( (float)(sizeof(mean3x3)) / sizeof(mean3x3[0][0]))) / 2;

	for(i=margin; i<height-margin; i++)
		for (j = margin; j < width - margin; j++)
		{
			mean = 0.0f;
			for (int m = -margin; m <= margin; m++)
				for (int n = -margin; n <= margin; n++)
					mean += (int)image[i + m][j + n] * mean3x3[m + margin][n + margin];;


			out_image[i][j] = (unsigned char)Saturation(mean);
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

	for(i=1; i<height-1; i++)
		for (j = 1; j < width-1; j++)
		{
			pidx = 0;

			for (m = -1; m <= 1; m++)
				for (n = -1; n <= 1; n++)
					pixel[pidx++] = (int)image[i +m ][j + n];

			int subi, subj, tmp;

			for(subi = 0; subi <NUM-1; subi++)
				for (subj = subi + 1; subj < NUM; subj++)
					if(pixel[subi]>pixel[subj])
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

void CImageProToolView::OnTwoAssignmentOneStretching()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();
	ASSERT_VALID(pDoc);

	struct STRPtr {
		BYTE r;
		BYTE g;
		BYTE b;
	};

	int i, j;

	int width = pDoc->m_Width;
	int height = pDoc->m_Height;


	BYTE* pData = pDoc->m_pDib->GetBitsAddress();
	RGBptr** ptr1 = new RGBptr*[height];
	STRPtr** image = new STRPtr*[height];

	for (i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		image[i] = new STRPtr[width];
	}

	Seperate_RGB(pData, ptr1);

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			image[i][j].r = Saturation(ptr1[i][j].r);
			image[i][j].g = Saturation(ptr1[i][j].g);
			image[i][j].b = Saturation(ptr1[i][j].b);
		}

	int max[3] = { 0 }, min[3] = { 0 }, Nmax, Nmin;
	Nmax = 255; Nmin = 0;
	int histoR[256] = { 0 };
	int histoG[256] = { 0 };
	int histoB[256] = { 0 };

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			histoR[image[i][j].r]++;
			histoG[image[i][j].g]++;
			histoB[image[i][j].b]++;
		}

	int count = 0;

	for (i = 0; i < 256; i++)
	{
		count += histoR[i];

		if (count > 1)
		{
			min[0] = i;
			break;
		}

	}
	count = 0;

	for (i = 255; i >= 0; i--)
	{
		count += histoR[i];

		if (count > 1)
		{
			max[0] = i;
			break;
		}
	}
	count = 0;

	for (i = 0; i < 256; i++)
	{
		count += histoG[i];

		if (count > 1)
		{
			min[1] = i;
			break;
		}

	}
	count = 0;

	for (i = 255; i >= 0; i--)
	{
		count += histoG[i];

		if (count > 1)
		{
			max[1] = i;
			break;
		}
	}

	count = 0;

	for (i = 0; i < 256; i++)
	{
		count += histoB[i];

		if (count > 1)
		{
			min[2] = i;
			break;
		}

	}
	count = 0;

	for (i = 255; i >= 0; i--)
	{
		count += histoB[i];

		if (count > 1)
		{
			max[2] = i;
			break;
		}
	}

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
//			if (image[i][j].r < min[0]) image[i][j].r = min[0];	// image[i][i]의 화소가 min 미만일때 min으로 보정
//			if (image[i][j].r > max[0]) image[i][j].r = max[0];	// image[i][j]의 화소가 max 이상일때 max으로 보정
			image[i][j].r = (int)((float)(image[i][j].r - min[0]) * ((float)(Nmax - Nmin) / (float)(max[0] - min[0])) + Nmin);

//			if (image[i][j].g < min[1]) image[i][j].g = min[1];	// image[i][i]의 화소가 min 미만일때 min으로 보정
//			if (image[i][j].g > max[1]) image[i][j].g = max[1];	// image[i][j]의 화소가 max 이상일때 max으로 보정
			image[i][j].g = (int)((float)(image[i][j].g - min[1]) * ((float)(Nmax - Nmin) / (float)(max[1] - min[1])) + Nmin);

//			if (image[i][j].b < min[2]) image[i][j].b = min[2];	// image[i][i]의 화소가 min 미만일때 min으로 보정
//			if (image[i][j].b > max[2]) image[i][j].b = max[2];	// image[i][j]의 화소가 max 이상일때 max으로 보정
			image[i][j].b = (int)((float)(image[i][j].b - min[2]) * ((float)(Nmax - Nmin) / (float)(max[2] - min[2])) + Nmin);
		}
	}

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			ptr1[i][j].r = image[i][j].r;
			ptr1[i][j].g = image[i][j].g;
			ptr1[i][j].b = image[i][j].b;
		}

	SetRGBptr(pData, ptr1, width, height);

	for (i = 0; i < height; i++)
	{
		delete[] ptr1[i];
		delete[] image[i];
	}

	delete[] ptr1;
	delete[] image;
	Invalidate();
}

void CImageProToolView::OnTwoassignmentOneEqaul()
{
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();
	ASSERT_VALID(pDoc);

	struct STRPtr {
		BYTE r;
		BYTE g;
		BYTE b;
	};

	int width = pDoc->m_Width;
	int height = pDoc->m_Height;

	BYTE* pData = pDoc->m_pDib->GetBitsAddress();

	RGBptr** ptr1 = new RGBptr*[height];
	STRPtr** image = new STRPtr*[height];

	for (int i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		image[i] = new STRPtr[width];
	}


	Seperate_RGB(pData, ptr1);

	int i, j, index;

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			image[i][j].r = ptr1[i][j].r;
			image[i][j].g = ptr1[i][j].g;
			image[i][j].b = ptr1[i][j].b;
		}

	float cumulation_r = 0.f;
	float cumulation_g = 0.f;
	float cumulation_b = 0.f;

	int histogram_r[256];
	int histogram_g[256];
	int histogram_b[256];
	memset(histogram_r, 0, sizeof(int) * 256);
	memset(histogram_g, 0, sizeof(int) * 256);
	memset(histogram_b, 0, sizeof(int) * 256);

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			histogram_r[image[i][j].r]++;
			histogram_g[image[i][j].g]++;
			histogram_b[image[i][j].b]++;
		}

	for (index = 0; index < 256; index++)
	{
		cumulation_r += (float)histogram_r[index];
		cumulation_g += (float)histogram_g[index];
		cumulation_b += (float)histogram_b[index];
		histogram_r[index] = (int)((cumulation_r / (float)(width*height))*255.0 + 0.5);
		histogram_g[index] = (int)((cumulation_g / (float)(width*height))*255.0 + 0.5);
		histogram_b[index] = (int)((cumulation_b / (float)(width*height))*255.0 + 0.5);
	}

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			index = image[i][j].r;
			image[i][j].r = histogram_r[index];
		}
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			index = image[i][j].g;
			image[i][j].g = histogram_g[index];
		}
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			index = image[i][j].b;
			image[i][j].b = histogram_b[index];
		}

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			ptr1[i][j].r = image[i][j].r;
			ptr1[i][j].g = image[i][j].g;
			ptr1[i][j].b = image[i][j].b;
		}


	SetRGBptr(pData, ptr1, width, height);

	for (i = 0; i < height; i++)
	{
		delete[] ptr1[i];
		delete[] image[i];
	}
	delete[] ptr1;
	delete[] image;

	Invalidate();
}

void CImageProToolView::OnTwoAssignmentTwo()
{
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();	// 불러온 문서변수 생성
	ASSERT_VALID(pDoc);

	int width = pDoc->m_Width;	// 불러온 문서의 너비추출
	int height = pDoc->m_Height;// 불러온 문서의 높이추출

	BYTE* pData = pDoc->m_pDib->GetBitsAddress();	// 문서의 화소주소값을 pData에 저장

	RGBptr** ptr1 = new RGBptr*[height];	//pData의 값을 복사할 **ptr1 생성
	BYTE** image = new BYTE*[height];	// ptr1의 값을 조작할 때 저장할 **image생성

										/* **가 가르치는 *의 값 생성*/
	for (int i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		image[i] = new BYTE[width];
	}

	Seperate_RGB(pData, ptr1);	// pData의 값을 **ptr1에 저장

	int i, j;	// for문을 위한 i,j변수 생성

	/*각 화소별 Gray 값에대한 채도(0-255범위지정)*/
	for (i = 0; i<height; i++)
		for (j = 0; j<width; j++)
			image[i][j] = Saturation((ptr1[i][j].r + ptr1[i][j].g + ptr1[i][j].b) / 3);

	int max, min, Nmax = 255, Nmin = 0; // 화소 밝기의 최대값과 최소값을 저장할 max, min 변수 생성
										// 스트레칭할 범위의 최대값과 최소값을 지정할 Nmax, Nmin 변수 생성
	int histogram[256];
	memset(&histogram, 0, sizeof(int) * 256);

	/*Saturation된 image의 Histogram 생성*/
	for (i = 0; i<height; i++)
		for (j = 0; j<width; j++)
			histogram[image[i][j]]++;


	int scaleValue = (int)((float)height*(float)width*0.05f); // 5퍼센트 Scaling 값 계산

	int count = 0;
	/*5퍼센트 개수보다 많이 count되면 min 값으로 저장*/
	for (i = 0; i<256; i++)
	{
		count += histogram[i];
		if (count>scaleValue)
		{
			min = i;
			break;
		}
	}
	count = 0; // count 초기화
	/*5퍼센트 개수보다 많이 count되면 max 값으로 저장*/
	for (i = 255; i >= 0; i--)
	{
		count += histogram[i];
		if (count>scaleValue)
		{
			max = i;
			break;
		}
	}

	for (i = 0; i<height; i++)
		for (j = 0; j<width; j++)
		{
			if (image[i][j] < min) image[i][j] = min;	// image[i][i]의 화소가 min 미만일때 min으로 보정
			if (image[i][j] > max) image[i][j] = max;	// image[i][j]의 화소가 max 이상일때 max으로 보정

			image[i][j] = (int)((float)(image[i][j] - min)*((float)(Nmax - Nmin) / (float)(max - min)) + Nmin);	// Stretching 계산
		}
	/*Stretching된 Gray기반의 image의 각 화소 값을 ptr1의 r,g,b에 저장*/
	for (i = 0; i<height; i++)
		for (j = 0; j<width; j++)
		{
			ptr1[i][j].r = image[i][j];
			ptr1[i][j].g = image[i][j];
			ptr1[i][j].b = image[i][j];
		}

	SetRGBptr(pData, ptr1, width, height);	// pData에 ptr1값을 저장(width, height 크기)

	/* double pointer 변수가 가르치는 pointer 변수 소멸자 호출*/
	for (i = 0; i<height; i++)
	{
		delete[] ptr1[i];
		delete[] image[i];
	}
	/*소멸자 호출*/
	delete[] ptr1;
	delete[] image;

	Invalidate();	// 출력화면을 갱신
}


void CImageProToolView::OnTwoAssignmentThree()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
}




void CImageProToolView::OnHistogramequal()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();	// 불러온 문서변수 생성
	ASSERT_VALID(pDoc);

	int width = pDoc->m_Width;	// 불러온 문서의 너비추출
	int height = pDoc->m_Height;// 불러온 문서의 높이추출

	BYTE* pData = pDoc->m_pDib->GetBitsAddress();	// 문서의 화소주소값을 pData에 저장

	RGBptr** ptr1 = new RGBptr*[height];	//pData의 값을 복사할 **ptr1 생성
	BYTE** image = new BYTE*[height];	// ptr1의 값을 조작할 때 저장할 **image생성

										/* double pointer 가 가르치는 *의 값 생성*/
	for (int i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		image[i] = new BYTE[width];
	}

	Seperate_RGB(pData, ptr1);	// pData의 값을 **ptr1에 저장

	int i, j;	// for문을 위한 i,j변수 생성
	int index;
	/*각 화소별 Gray 값에대한 채도(0-255범위지정)*/
	for (i = 0; i<height; i++)
		for (j = 0; j<width; j++)
			image[i][j] = Saturation((ptr1[i][j].r + ptr1[i][j].g + ptr1[i][j].b) / 3);

	float cumulation = 0.0;	// //
	int histogram[256];	// Histogram 생성
	memset(histogram, 0, sizeof(int) * 256); //histogram 0으로 초기화


	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			histogram[image[i][j]]++;

	for (index = 0; index < 256; index++)
	{
		cumulation += (float)histogram[index]; // 히스토그램 누적값

		histogram[index] = (int)((cumulation / (float)(width*height)) * 255 + 0.5); // 누적 히스토그램에 대한 정규화(0 ... 1) AND 255로 확장
	}

	for(i=0; i<height; i++)
		for (j = 0; j < width; j++)
		{
			index = image[i][j];	// 해당 픽셀의 화소 값 저장
			image[i][j] = histogram[index];	// 해당 화소 값에 대응되는 T(r)값 적용
		}


	for (i = 0; i<height; i++)
		for (j = 0; j<width; j++)
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
	}
	delete[] ptr1;
	delete[] image;

	Invalidate();
}
