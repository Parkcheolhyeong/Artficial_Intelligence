// ImageProToolView.cpp : implementation of the CImageProToolView class
// Git master
// GIt hotfix

#pragma warning(disable: 4996) 
#define _CRT_SECURE_NO_DEPRECATE

/*헤더파일 호출*/
#include "stdafx.h"
#include "ImageProTool.h"
#include "Dib.h"
#include "math.h"
#include "time.h"
#include "ImageProToolDoc.h"
#include "ImageProToolView.h"
#include "Histogram.h"
#include "NeuralNet.h"

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
	ON_COMMAND(ID_TERMPROJECT_TEST, &CImageProToolView::OnTest)
	ON_COMMAND(ID_TERMPROJECT_TRAIN, &CImageProToolView::OnTrain)
	ON_COMMAND(ID_TRAINMODE_0, &CImageProToolView::OnTrainmode0)
	ON_COMMAND(ID_TRAINMODE_1, &CImageProToolView::OnTrainmode1)
	ON_COMMAND(ID_TRAINMODE_2, &CImageProToolView::OnTrainmode2)
	ON_COMMAND(ID_TRAINMODE_4, &CImageProToolView::OnTrainmode4)
	ON_COMMAND(ID_TRAINMODE_5, &CImageProToolView::OnTrainmode5)
	ON_COMMAND(ID_TRAINMODE_6, &CImageProToolView::OnTrainmode6)
	ON_COMMAND(ID_TRAINMODE_7, &CImageProToolView::OnTrainmode7)
	ON_COMMAND(ID_TRAINMODE_8, &CImageProToolView::OnTrainmode8)
	ON_COMMAND(ID_TRAINMODE_9, &CImageProToolView::OnTrainmode9)
	ON_COMMAND(ID_TRAINMODE_3, &CImageProToolView::OnTrainmode3)
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


void CImageProToolView::OnTest()
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

	NeuronD input[784];

	Neuron my_neuron;

	//initialize
	my_neuron.init();
	my_neuron.loadWeight();
	int cnt = 0;
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			input[cnt++] = ptr1[i][j].r / 255.0;
		}
	my_neuron.propForward(input);

	my_neuron.print();

	
	my_neuron.fin();
	for (i = 0; i < height; i++)
	{
		delete[] ptr1[i];
	}

	delete[] ptr1;
}


void CImageProToolView::OnTrain()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
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

	NeuronD input[784];

	Neuron my_neuron;

	//initialize
	my_neuron.init();
	my_neuron.loadWeight();
	int cnt = 0;

	for (i = height - 1; i >= 0; i--)
		for (j = width - 1; j >= 0; j--)
		{
			input[cnt++] = ptr1[i][j].r / 255.0;
		}
	my_neuron.propForward(input);

	my_neuron.print();


	my_neuron.fin();
	for (i = 0; i < height; i++)
	{
		delete[] ptr1[i];
	}

	delete[] ptr1;
}

void CImageProToolView::trainSet(int k)
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();
	ASSERT_VALID(pDoc);

	int width = pDoc->m_Width;
	int height = pDoc->m_Height;
	int i, j;
	char file_name[100];

	BYTE* pData = pDoc->m_pDib->GetBitsAddress();
	RGBptr** ptr1 = new RGBptr*[height];

	for (i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
	}

	Seperate_RGB(pData, ptr1);

	NeuronD input[784];

	Neuron neuronSet;

	//initialize
	neuronSet.init();
	neuronSet.loadWeight();
	//training

	NeuronD target[10] = { 0, };
	target[k] = 1;

	CString str;
	str.Format(_T("%d번 학습시도"), k);
	AfxMessageBox(str);

	int cnt = 0;
	for (i = height - 1; i >= 0; i--)
		for (j = width - 1; j >= 0; j--)
		{
			input[cnt++] = ptr1[i][j].r / 255.0;
		}
	neuronSet.propForward(input);
	neuronSet.propBackward(target);

	neuronSet.saveWeight();
	//end
	neuronSet.fin();

	for (i = 0; i < height; i++)
	{
		delete[] ptr1[i];
	}

	delete[] ptr1;
}

void CImageProToolView::OnTrainmode0()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	trainSet(0);
}

void CImageProToolView::OnTrainmode1()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	trainSet(1);
}

void CImageProToolView::OnTrainmode2()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	trainSet(2);
}


void CImageProToolView::OnTrainmode4()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	trainSet(3);
}

void CImageProToolView::OnTrainmode3()

{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	trainSet(4);
}

void CImageProToolView::OnTrainmode5()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	trainSet(5);
}

void CImageProToolView::OnTrainmode6()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	trainSet(6);
}

void CImageProToolView::OnTrainmode7()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	trainSet(7);
}

void CImageProToolView::OnTrainmode8()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	trainSet(8);
}

void CImageProToolView::OnTrainmode9()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	trainSet(9);
}
