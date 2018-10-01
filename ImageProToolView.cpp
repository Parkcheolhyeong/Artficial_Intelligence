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
	ON_COMMAND(ID_CHAP_MEANFILTERING, &CImageProToolView::OnMeanfiltering)
	ON_COMMAND(ID_CHAP_MEDIANFILTERING, &CImageProToolView::OnMedianfiltering)
	ON_COMMAND(ID_ASSIGNMENT_TWO, &CImageProToolView::OnTwoAssignmentTwo)
	ON_COMMAND(ID_ASSIGNMENT_ONE, &CImageProToolView::OnTwoAssignmentOneStretching)
	ON_COMMAND(ID_ASSIGNMENT_THREE, &CImageProToolView::OnTwoAssignmentThree)
	ON_COMMAND(ID_CHAPTWOASSIGNMENT_TWOEQAUL, &CImageProToolView::OnTwoassignmentOneEqaul)
	ON_COMMAND(ID_CHAP_HISTOGRAM32884, &CImageProToolView::OnHistogram)
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




void CImageProToolView::OnTwoAssignmentOneStretching()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();	// 파일로드
	ASSERT_VALID(pDoc);	// pDoc의 유효성 검사

	struct STRPtr {	// R, G, B 각각 계산을 위해 구조체 선언
		BYTE r;
		BYTE g;
		BYTE b;
	};

	int i, j;	// for문을 위한 변수선언

	int width = pDoc->m_Width;	// 파일의 너비 저장
	int height = pDoc->m_Height;// 파일의 높이 저장


	BYTE* pData = pDoc->m_pDib->GetBitsAddress(); // 파일을 가르치는 bit의 주소값 저장
	RGBptr** ptr1 = new RGBptr*[height];  // 파일의 각 화소값을 저장할 double pointer 변수 생성
	STRPtr** image = new STRPtr*[height]; // 파일의 R, G, B 조작 값을 저장할 double pointer 변수 생성

	/*각 double pointer 변수 별로 pointer 선언*/
	for (i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		image[i] = new STRPtr[width];
	}

	Seperate_RGB(pData, ptr1); // pData의 값을 ptr1에 저장

	/* R, G, B 세 가지 채널에 대한 계산 및 pixel의 밝기를 0-255범위를 가지도록 설정 후 저장 */
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			image[i][j].r = Saturation(ptr1[i][j].r);
			image[i][j].g = Saturation(ptr1[i][j].g);
			image[i][j].b = Saturation(ptr1[i][j].b);
		}

	/* 화소 밝기의 최대값과 최소값을 저장할 배열 선언 및 초기화*/
	/* 원하는 스트레칭 값 범위 지정을 위한 변수 선언 */
	int max[3], min[3], Nmax, Nmin;
	memset(&max, 0, sizeof(int) * 3);
	memset(&min, 0, sizeof(int) * 3);

	Nmax = 255; Nmin = 0;	// 스트레칭 값 범위설정

	/*히스토그램배열 선언 및 초기화*/
	int histoR[256];
	int histoG[256];
	int histoB[256];
	memset(&histoR, 0, sizeof(int) * 256);
	memset(&histoG, 0, sizeof(int) * 256);
	memset(&histoB, 0, sizeof(int) * 256);

	/*R, G, B 채널에대한 히스토그램 계산*/
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			histoR[image[i][j].r]++;
			histoG[image[i][j].g]++;
			histoB[image[i][j].b]++;
		}

	int count = 0; // 화소의 최대와 최소 지점 계산을 위한 변수 선언

	/* R, G, B 에대한 화소의 최대와 최소의 지점 계산 */
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


	/* R, G, B 각각 탐색한 min과 max의 범위를 Nmax와 Nmin을 활용해 스트레칭계산 */
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (image[i][j].r < min[0]) image[i][j].r = min[0]; if (image[i][j].r > max[0]) image[i][j].r = max[0];
			if (image[i][j].r < min[1]) image[i][j].r = min[1]; if (image[i][j].r > max[1]) image[i][j].r = max[1];
			if (image[i][j].r < min[2]) image[i][j].r = min[2]; if (image[i][j].r > max[2]) image[i][j].r = max[2];

			image[i][j].r = (int)((float)(image[i][j].r - min[0]) * ((float)(Nmax - Nmin) / (float)(max[0] - min[0])) + Nmin);
			image[i][j].g = (int)((float)(image[i][j].g - min[1]) * ((float)(Nmax - Nmin) / (float)(max[1] - min[1])) + Nmin);
			image[i][j].b = (int)((float)(image[i][j].b - min[2]) * ((float)(Nmax - Nmin) / (float)(max[2] - min[2])) + Nmin);
		}
	}

	/*r,g,b 채널 별로 ptr1에 저장*/
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			ptr1[i][j].r = image[i][j].r;
			ptr1[i][j].g = image[i][j].g;
			ptr1[i][j].b = image[i][j].b;
		}

	/*ptr1 값을 pData에 저장*/
	SetRGBptr(pData, ptr1, width, height);


	/*소멸자 호출*/
	for (i = 0; i < height; i++)
	{
		delete[] ptr1[i];
		delete[] image[i];
	}

	delete[] ptr1;
	delete[] image;
	Invalidate();	// 화면 업데이트
}

void CImageProToolView::OnTwoassignmentOneEqaul()
{
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();	// 파일로드
	ASSERT_VALID(pDoc);	// pDoc 유효성검사

	/* R, G, B 각각 계산을 위해 구조체 선언*/
	struct STRPtr {	
		BYTE r;
		BYTE g;
		BYTE b;
	};

	int i, j, index;	//for문 및 누적히스토그램 계산을 위한 변수 선언
	int width = pDoc->m_Width;	// 파일의 너비 저장
	int height = pDoc->m_Height;// 파일의 높이 저장


	BYTE* pData = pDoc->m_pDib->GetBitsAddress(); // 파일을 가르치는 bit의 주소값 저장
	RGBptr** ptr1 = new RGBptr*[height];  // 파일의 각 화소값을 저장할 double pointer 변수 생성
	STRPtr** image = new STRPtr*[height]; // 파일의 R, G, B 조작 값을 저장할 double pointer 변수 생성

	/*각 double pointer 변수 별로 pointer 선언*/
	for (int i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		image[i] = new STRPtr[width];
	}


	Seperate_RGB(pData, ptr1); // pData의 값을 ptr1에 저장

	/* R, G, B 세 가지 채널에 대한 계산 및 pixel의 밝기를 0-255범위를 가지도록 설정 후 저장 */
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			image[i][j].r = Saturation(ptr1[i][j].r);
			image[i][j].g = Saturation(ptr1[i][j].g);
			image[i][j].b = Saturation(ptr1[i][j].b);
		}

	/* 누적히스토그램을 위한 변수 및 히스토그램을 위한 배열 선언 및 초기화*/
	float cumulationR = 0.f;
	float cumulationG = 0.f;
	float cumulationB = 0.f;

	int histoR[256];
	int histoG[256];
	int histoB[256];
	memset(histoR, 0, sizeof(int) * 256);
	memset(histoG, 0, sizeof(int) * 256);
	memset(histoB, 0, sizeof(int) * 256);

	/*R, G, B 채널에 대한 히스토그램 생성*/
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			histoR[image[i][j].r]++;
			histoG[image[i][j].g]++;
			histoB[image[i][j].b]++;
		}

	/* 누적되는 값에대한 정규화(0~1) 및 255로 확장(반올림을 위해 +0.5) */
	for (index = 0; index < 256; index++)
	{
		cumulationR += (float)histoR[index];
		cumulationG += (float)histoG[index];
		cumulationB += (float)histoB[index];
		histoR[index] = (int)((cumulationR / (float)(width*height))*255.0 + 0.5);
		histoG[index] = (int)((cumulationG / (float)(width*height))*255.0 + 0.5);
		histoB[index] = (int)((cumulationB / (float)(width*height))*255.0 + 0.5);
	}

	/* 입력 pixel에 대응되는 T(r)값 적용*/
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			index = image[i][j].r;
			image[i][j].r = histoR[index];
		}
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			index = image[i][j].g;
			image[i][j].g = histoG[index];
		}
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			index = image[i][j].b;
			image[i][j].b = histoB[index];
		}

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			ptr1[i][j].r = image[i][j].r;
			ptr1[i][j].g = image[i][j].g;
			ptr1[i][j].b = image[i][j].b;
		}

	/*ptr1 값을 pData에 저장*/
	SetRGBptr(pData, ptr1, width, height);

	/*소멸자 호출*/
	for (i = 0; i < height; i++)
	{
		delete[] ptr1[i];
		delete[] image[i];
	}
	delete[] ptr1;
	delete[] image;

	Invalidate();	// 출력화면 갱신
}

void CImageProToolView::OnTwoAssignmentTwo()
{
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();	// 문서로드
	ASSERT_VALID(pDoc);	// pDoc 유효성 확인

	int width = pDoc->m_Width;	// 불러온 문서의 너비추출
	int height = pDoc->m_Height;// 불러온 문서의 높이추출

	BYTE* pData = pDoc->m_pDib->GetBitsAddress();	// 문서의 화소주소값을 pData에 저장

	RGBptr** ptr1 = new RGBptr*[height];	//pData의 값을 복사할 **ptr1 생성
	BYTE** image = new BYTE*[height];	// ptr1의 값을 조작할 때 저장할 **image생성

	/* pointer 값 선언 및 초기화*/
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


	int scaleValue = (int)((float)height*(float)width*0.3f); // 5퍼센트 Scaling 값 계산

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

	/*소멸자 호출*/
	for (i = 0; i<height; i++)
	{
		delete[] ptr1[i];
		delete[] image[i];
	}

	delete[] ptr1;
	delete[] image;

	Invalidate();	// 출력화면을 갱신
}


void CImageProToolView::OnTwoAssignmentThree()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
}

void CImageProToolView::OnHistogram()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
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


	//Histogram* dlg = new Histogram(); //객체 포인터
	//dlg->SetImage(histogray, histoR, histoG, histoB);
	//dlg->DoModal();

	Histogram dlg; // 객체변수
	dlg.SetImage(histogray, histoR, histoG, histoB);
	dlg.DoModal();

}
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
	BYTE** image = new BYTE*[height];				 // **ptr1의 조작결과를 저장시킬 double pointer 변수 생성
	BYTE** out_image = new BYTE*[height];			 // **image의 조작결과를 저장시킬 double pointer 변수 생성

													 /*각 double pointer 변수 별로 pointer 선언*/
	for (i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		image[i] = new BYTE[width];
		out_image[i] = new BYTE[width];
	}

	Seperate_RGB(pData, ptr1);	// pData의 값을 ptr1에 저장

								/* Gray값 계산 및 pixel의 밝기를 0-255범위를 가지도록 설정 후 image 변수에 저장 */
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			image[i][j] = (BYTE)Saturation((ptr1[i][j].r + ptr1[i][j].g + ptr1[i][j].b) / 3);

	float mean = 0.f;	// 평균 값을 저장할 변수선언

						/*3x3 크기의 mask 선언 및 초기화*/
	float mean3x3[3][3] = { 1.0f / 9.0f, 1.0f / 9.0f , 1.0f / 9.0f,
		1.0f / 9.0f , 1.0f / 9.0f  ,1.0f / 9.0f,
		1.0f / 9.0f, 1.0f / 9.0f , 1.0f / 9.0f };

	int margin;	// 여백크기를 저장할 변수선언

	margin = sqrt(((float)(sizeof(mean3x3)) / sizeof(mean3x3[0][0]))) / 2;	// mask값에 해당하는 여백 계산
																			// sqrt는 루트계산을 의미

																			/* 시작을 여백을 포함한 i부터 여백을 제외한 heigth-margin까지 mask 슬라이딩*/
	for (i = margin; i<height - margin; i++)
		for (j = margin; j < width - margin; j++)
		{
			mean = 0.0f;	// mean 값 초기화

							/*image의 -margin부터 margin까지 탐색하며 image와 mask에 각 계수들로 mean계산*/
			for (int m = -margin; m <= margin; m++)
				for (int n = -margin; n <= margin; n++)
					mean += (int)image[i + m][j + n] * mean3x3[m + margin][n + margin];;


			out_image[i][j] = (unsigned char)Saturation(mean);	// image로 계산된 mean값을 채도설정 후 out_image로 다시 저장한다.
		}

	/*mean-filtering이 완료된 값을 ptr1에 저장*/
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			ptr1[i][j].r = ptr1[i][j].g = ptr1[i][j].b = out_image[i][j];

	/*ptr1값을 pData에 저장*/
	SetRGBptr(pData, ptr1, width, height);

	/*소멸자 호출*/
	for (i = 0; i < height; i++)
	{
		delete[] ptr1[i];
		delete[] image[i];
		delete[] out_image[i];
	}

	delete[] ptr1;
	delete[] out_image;
	delete[] image;

	Invalidate(); //	출력화면 갱신
}


void CImageProToolView::OnMedianfiltering()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();	// 파일로드
	ASSERT_VALID(pDoc);	// pDoc 유효성검사
	int i, j, m, n; // for문 및 mask를 위한 변수선언

	int width = pDoc->m_Width;	// 파일의 너비 저장
	int height = pDoc->m_Height;// 파일의 높이 저장

	BYTE* pData = pDoc->m_pDib->GetBitsAddress();// 파일의 픽셀을 가르치는 pointer 변수 생성
	RGBptr** ptr1 = new RGBptr*[height];	   	 // 파일의 각 화소값을 저장할 double pointer 변수 생성
	BYTE** image = new BYTE*[height];				 // **ptr1의 조작결과를 저장시킬 double pointer 변수 생성
	BYTE** out_image = new BYTE*[height];			 // **image의 조작결과를 저장시킬 double pointer 변수 생성

													 /*각 double pointer 변수 별로 pointer 선언*/
	for (i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		image[i] = new BYTE[width];
		out_image[i] = new BYTE[width];
	}


	Seperate_RGB(pData, ptr1);	// pData의 값을 ptr1에 저장

								/* Gray값 계산 및 pixel의 밝기를 0-255범위를 가지도록 설정 후 image 변수에 저장 */
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			image[i][j] = (BYTE)Saturation((ptr1[i][j].r + ptr1[i][j].g + ptr1[i][j].b) / 3);

	int median = 0;	// median 값 저장 변수 선언

	int NUM = 9;	// mask 크기 저장
	int* pixel = new int[NUM];	// 화소값을 저장할 pointer pixel 변수 선언 및 초기화(mask 크기)
	int pidx;	// *pixel 카운트를 위한 변수 선언

				/*mask의 크기를 미리알고 있기에 margin 값인 1부터 for문 실행*/
	for (i = 1; i<height - 1; i++)
		for (j = 1; j < width - 1; j++)
		{
			pidx = 0;

			for (m = -1; m <= 1; m++)
				for (n = -1; n <= 1; n++)
					pixel[pidx++] = (int)image[i + m][j + n];

			int subi, subj, tmp;	// 정렬을 위한 변수 선언

									/* 오름차순 정렬 */
			for (subi = 0; subi <NUM - 1; subi++)
				for (subj = subi + 1; subj < NUM; subj++)
					if (pixel[subi]>pixel[subj])
					{
						tmp = pixel[subi];
						pixel[subi] = pixel[subj];
						pixel[subj] = tmp;
					}

			/* NUM이 홀수인 경우 이전의 값과 더해서 평균계산 후 적용 */
			if (NUM % 2 != 0)
				median = pixel[NUM / 2];
			else
				median = (pixel[NUM / 2 - 1] + pixel[NUM / 2]) / 2;

			// 채도 설정후 out_image에 저장
			out_image[i][j] = (unsigned char)Saturation(median);
		}

	// out_image 값을 ptr1에 저장
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			ptr1[i][j].r = ptr1[i][j].g = ptr1[i][j].b = out_image[i][j];

	//ptr1 값을 pData에 저장
	SetRGBptr(pData, ptr1, width, height);

	/* 소멸자 호출 */
	for (i = 0; i < height; i++)
	{
		delete[] ptr1[i];
		delete[] image[i];
		delete[] out_image[i];
	}

	delete[] ptr1;
	delete[] out_image;
	delete[] image;
	Invalidate();	//출력화면 업데이트
}