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
	ON_COMMAND(ID_RGBTOGRAY, &CImageProToolView::OnRgbtogray)
	ON_COMMAND(ID_HISTOGRAM, &CImageProToolView::OnHistogram)
	ON_COMMAND(ID_CHAP_THRESHOLDING, &CImageProToolView::OnChapThresholding)
	ON_COMMAND(ID_CHAP_GLOBALTHRESHOLDING, &CImageProToolView::OnGlobalthresholding)
	ON_COMMAND(ID_CHAP_HISTOGRAMSTRETCHING, &CImageProToolView::OnHistogramstretching)
	ON_COMMAND(ID_CHAP_MEANFILTERING, &CImageProToolView::OnMeanfiltering)
	ON_COMMAND(ID_CHAP_MEDIANFILTERING, &CImageProToolView::OnMedianfiltering)
	ON_COMMAND(ID_ASSIGNMENT_TWO, &CImageProToolView::OnTwoAssignmentTwo)
	ON_COMMAND(ID_ASSIGNMENT_ONE, &CImageProToolView::OnTwoAssignmentOneStretching)
	ON_COMMAND(ID_ASSIGNMENT_THREE, &CImageProToolView::OnTwoAssignmentThree)
	ON_COMMAND(ID_CHAPTWOASSIGNMENT_TWOEQAUL, &CImageProToolView::OnTwoassignmentOneEqaul)
	ON_COMMAND(ID_CHAP_SOBEL, &CImageProToolView::OnSobel)
	ON_COMMAND(ID_CHAP_NEARESTSCALING, &CImageProToolView::OnNearestscaling)
	ON_COMMAND(ID_CHAP_BINARYDILATION, &CImageProToolView::OnBinarydilation)
	ON_COMMAND(ID_CHAP_ROTATION, &CImageProToolView::OnRotation)
	ON_COMMAND(ID_CHAP_BINARYEROSION, &CImageProToolView::OnBinaryerosion)
	ON_COMMAND(ID_CHAP_CONNECTEDLABELING, &CImageProToolView::OnConnectedlabeling)
	ON_COMMAND(ID_CHAP_ONINVARIANTMOMENT, &CImageProToolView::Oninvariantmoment)
	ON_COMMAND(ID_CHAP_K, &CImageProToolView::OnKmeans)
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


void CImageProToolView::OnRgbtogray()
{
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();
	ASSERT_VALID(pDoc);

	int width = pDoc->m_Width;
	int height = pDoc->m_Height;	
 	
	BYTE* pData = pDoc->m_pDib->GetBitsAddress();
	RGBptr** ptr1= new RGBptr*[height];
	  
	for( int i = 0 ; i < height ; i++ )
	{
	      ptr1[i]= new RGBptr [width];		 
	 }		
	
   	Seperate_RGB(pData,ptr1); 
	BYTE gray;
	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++) 	
		{
			gray=(BYTE)Saturation((ptr1[i][j].r+ptr1[i][j].g+ptr1[i][j].b)/3);
			ptr1[i][j].r = ptr1[i][j].g = ptr1[i][j].b = gray;
		}
	}

	//화면 출력을 위한 과정
	
	SetRGBptr(pData,ptr1,width,height);

	for(int i = 0 ; i < height ; i++ )
	{
		delete [] ptr1[i];
	}
	delete [] ptr1;	
		
    Invalidate();
}


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


	//Histogram* dlg = new Histogram(); //객체 포인터
	//dlg->SetImage(histogray, histoR, histoG, histoB);
	//dlg->DoModal();

	Histogram dlg; // 객체변수
	dlg.SetImage(histogray, histoR, histoG, histoB);
	dlg.DoModal();
}


void CImageProToolView::OnChapThresholding()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();
	ASSERT_VALID(pDoc);

	int width = pDoc->m_Width;
	int height = pDoc->m_Height;
	int i, j;

	BYTE* pData = pDoc->m_pDib->GetBitsAddress();
	RGBptr** ptr1 = new RGBptr*[height];
	// BYTE** image = new BYTE*[height];

	for (int i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		// image[i] = new BYTE[width];
	}

	Seperate_RGB(pData, ptr1);
	int T1 = 120;
	int* Y = new int[height*width];

	memset(Y, 0, height*width);

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			Y[i*width + j] = Saturation((ptr1[i][j].r + ptr1[i][j].g + ptr1[i][j].b) / 3); // 그레이 영상으로 바꾸기
		}
	}

	// T1 임계값에 따라 이진화
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (Y[i*width + j] < T1) {
				ptr1[i][j].r = 255;
				ptr1[i][j].g = 255;
				ptr1[i][j].b = 255;
			}
			else {
				ptr1[i][j].r = 0;
				ptr1[i][j].g = 0;
				ptr1[i][j].b = 0;
			}
		}
	}

	SetRGBptr(pData, ptr1, width, height);
	// 메모리에 신경쓰기 위해서 강제적으로 없애서 메모리 관리
	for (int i = 0; i < height; i++)
	{
		delete[] ptr1[i];
	}
	delete[] ptr1;
	delete[] Y;
	Invalidate();
}


void CImageProToolView::OnGlobalthresholding()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();
	ASSERT_VALID(pDoc);
	int i, j, T1 = 1, T2 = 0, T0 = 1;

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

	for(i=0; i<height; i++)
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

			for(i=0; i<height; i++)
				for (j = 0; j < width; j++)
				{
					if (Y[i*width + j] > T2) Bi[i*width + j] = 255;
					else Bi[i*width + j] = 0;
				}
		}
	}

	for(i=0; i<height; i++)
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

void CImageProToolView::OnHistogramstretching()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();
	ASSERT_VALID(pDoc);
	int i, j;

	int width = pDoc->m_Width;
	int height = pDoc->m_Height;


	BYTE* pData = pDoc->m_pDib->GetBitsAddress();
	RGBptr** ptr1 = new RGBptr*[height];
	int** image = new int*[height];

	for (i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		image[i] = new int[width];
	}

	Seperate_RGB(pData, ptr1);

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			image[i][j] = Saturation((ptr1[i][j].r + ptr1[i][j].g + ptr1[i][j].b) / 3);


	int histogram[256];
	memset(histogram, 0, sizeof(int)*256);

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			histogram[image[i][j]]++;

	int max = histogram[0], min = histogram[0], Nmax, Nmin;
	Nmax = 255; Nmin = 0;
	int count = 0;

	for (i = 0; i < 256; i++)
	{
		count += histogram[i];
		if (count > 1)
		{
			min = i;
			break;
		}
	}

	count = 0;
	for (i = 255; i >= 0; i--)
	{
		count += histogram[i];
		if (count > 1)
		{
			max = i;
			break;
		}
	}

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++) {
			if (image[i][j] < min) image[i][j] = min; 
			if (image[i][j] > max) image[i][j] = max;
			
			image[i][j] = (int)((float)(image[i][j] - min) * ((float)(Nmax - Nmin) / (float)(max - min)) + Nmin);
		}
	}
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)

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
	}

	delete[] ptr1;
	delete[] image;
	Invalidate();
}
void CImageProToolView::OnMeanfiltering()
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
			for (m = -margin; m <= margin; m++)
				for (n = -margin; n <= margin; n++)
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

	/*각 화소별 Gray 값에대한 채도(0-255사이)계산*/
	for (i = 0; i<height; i++)
		for (j = 0; j<width; j++)
			image[i][j] = Saturation((ptr1[i][j].r + ptr1[i][j].g + ptr1[i][j].b) / 3);

	int max, min, Nmax = 255, Nmin = 0; // 화소 밝기의 최대값과 최소값을 저장할 max, min 변수 생성
										// 스트레칭할 범위의 최대값과 최소값을 지정할 Nmax, Nmin 변수 생성
	int histogram[256];
	memset(histogram, 0, sizeof(int) * 256);

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

	/*더블 포인터 변수가 가르치는 포인터 변수에 소멸자 호출*/
	for (i = 0; i<height; i++)
	{
		delete[] ptr1[i];
		delete[] image[i];
	}
	/*소멸자 호출*/
	delete[] ptr1;
	delete[] image;

	Invalidate();	// 갱신된 값을 표시하기위해 출력면을 갱신.
}
void CImageProToolView::OnTwoassignmentOneEqaul()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();
	ASSERT_VALID(pDoc);
	
	struct satuPtr {
		int r;
		int g;
		int b;
	};
	
	int i, j, index;

	int width = pDoc->m_Width;
	int height = pDoc->m_Height;


	BYTE* pData = pDoc->m_pDib->GetBitsAddress();
	RGBptr** ptr1 = new RGBptr*[height];
	satuPtr** image = new satuPtr*[height];

	for (i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		image[i] = new satuPtr[width];
	}

	Seperate_RGB(pData, ptr1);

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			image[i][j].r = Saturation(ptr1[i][j].r);
			image[i][j].g = Saturation(ptr1[i][j].g);
			image[i][j].b = Saturation(ptr1[i][j].b);
		}

	float cumulationR = 0.0;
	float cumulationG = 0.0;
	float cumulationB = 0.0;

	int histoR[256];
	int histoG[256];
	int histoB[256];
	memset(histoR, 0, sizeof(int) * 256);
	memset(histoG, 0, sizeof(int) * 256);
	memset(histoB, 0, sizeof(int) * 256);

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			histoR[image[i][j].r]++;
			histoG[image[i][j].g]++;
			histoB[image[i][j].b]++;
		}

	for (index = 0; index < 256; index++)
	{
		cumulationR += (float)histoR[index];
		cumulationG += (float)histoG[index];
		cumulationB += (float)histoB[index];

		histoR[index] = (int)((cumulationR / (float)(width*height)*255.0 + 0.5));
		histoG[index] = (int)((cumulationG / (float)(width*height)*255.0 + 0.5));
		histoB[index] = (int)((cumulationB / (float)(width*height)*255.0 + 0.5));

	}

	for (i = 0; i<height; i++)
		for (j = 0; j < width; j++)
		{
			index = image[i][j].r;
			image[i][j].r = histoR[index];
			image[i][j].g = histoG[index];
			image[i][j].b = histoB[index];
		}

	for (i = 0; i<height; i++)
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

void CImageProToolView::OnTwoAssignmentOneStretching()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();
	ASSERT_VALID(pDoc);

	struct STRPtr {
		int r;
		int g;
		int b;
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
	int histoR[256];
	int histoG[256];
	int histoB[256];

	memset(histoR, 0, sizeof(int) * 256);
	memset(histoG, 0, sizeof(int) * 256);
	memset(histoB, 0, sizeof(int) * 256);
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			histoR[image[i][j].r]++;
			histoG[image[i][j].g]++;
			histoB[image[i][j].b]++;
		}

	int count[3]; bool valCount[3];
	memset(count, 0, sizeof(int) * 3);
	memset(valCount, false, sizeof(bool) * 3);
	for (i = 0; i < 256; i++)
	{
		count[0] += histoR[i];
		count[1] += histoG[i];
		count[2] += histoB[i];

		if (count[0] > 1 && valCount[0] == false)
		{
			min[0] = i;
			valCount[0] = true;
			break;
		}
		else if (count[1] > 1 && valCount[0] == false)
		{
			min[1] = i;
			valCount[0] = true;
			break;
		}
		else if (count[2] > 1 && valCount[0] == false)
		{
			min[2] = i;
			valCount[0] = true;
			break;
		}
	}

	memset(count, 0, sizeof(int) * 3);
	memset(valCount, false, sizeof(bool) * 3);

	for (i = 255; i >= 0; i--)
	{
		count[0] += histoR[i];
		count[1] += histoG[i];
		count[2] += histoB[i];

		if (count[0] > 1)
		{
			max[0] = i;
			break;
		}
		else if (count[1] > 1)
		{
			max[1] = i;
			break;
		}
		else if (count[2] > 1)
		{
			max[2] = i;
			break;
		}
	}

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			image[i][j].r = (int)((float)(image[i][j].r - min[0]) * ((float)(Nmax - Nmin) / (float)(max[0] - min[0])) + Nmin);
			image[i][j].g = (int)((float)(image[i][j].g - min[1]) * ((float)(Nmax - Nmin) / (float)(max[1] - min[1])) + Nmin);
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


void CImageProToolView::OnTwoAssignmentThree()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
}




void CImageProToolView::OnSobel()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.


	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();	// 불러온 문서변수 생성
	ASSERT_VALID(pDoc);

	int i, j, m, n;

	int width = pDoc->m_Width;	// 불러온 문서의 너비추출
	int height = pDoc->m_Height;// 불러온 문서의 높이추출

	BYTE* pData = pDoc->m_pDib->GetBitsAddress();	// 문서의 화소주소값을 pData에 저장

	RGBptr** ptr1 = new RGBptr*[height];	//pData의 값을 복사할 **ptr1 생성
	BYTE** image = new BYTE*[height];	// ptr1의 값을 조작할 때 저장할 **image생성
	BYTE** out_image = new BYTE*[height];

	/* **가 가르치는 *의 값 생성*/
	for (int i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		image[i] = new BYTE[width];
		out_image[i] = new BYTE[width];
	}

	Seperate_RGB(pData, ptr1);	// pData의 값을 **ptr1에 저장

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			image[i][j] = (BYTE)Saturation((ptr1[i][j].r + ptr1[i][j].g + ptr1[i][j].b) / 3);

	int g1, g2, gradient;
	g1 = g2 = gradient = 0;

	int gx[3][3] = { -1, 0, 1,
					 -2, 0, 2,
					 -1, 0, 1 };
	int gy[3][3] = { 1, 2, 1,
					 0, 0, 0,
					 -1, -2, -1};

	for (i = 1; i < height-1; i++)
	{
		for (j = 1; j < width-1; j++)
		{
			for (m = -1; m <= 1; m++)
			{
				for (n = -1; n <= 1; n++)
				{
					g1 += (int)image[i + m][j + n] * gx[m + 1][n + 1];
					g2 += (int)image[i + m][j + n] * gy[m + 1][n + 1];
				}
			}
			gradient = Saturation(abs(g1) + abs(g2));
			out_image[i][j] = (unsigned char)gradient;
			g1 = g2 = 0;
		}
	}




	for (i = 0; i < height-1; i++)
		for (j = 0; j < width-1; j++)
		{
			ptr1[i][j].r = out_image[i][j];
			ptr1[i][j].g = out_image[i][j];
			ptr1[i][j].b = out_image[i][j];
		}

	SetRGBptr(pData, ptr1, width, height);

	for (i = 0; i < height; i++)
	{
		delete[] ptr1[i];
		delete[] image[i];
		delete[] out_image[i];
	}

	delete[] ptr1;
	delete[] image;
	delete[] out_image;

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
	double scale = 0.3;
	int new_height = (int)(height * scale);
	int new_width = (int)(width * scale);

	RGBptr **ptr2 = new RGBptr*[new_height];
	for (i = 0; i < new_height; i++)
		ptr2[i] = new RGBptr[new_width];

	for(i =0; i<new_height; i++)
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

		for(i=0; i<new_height; i++)
			for (j = 0; j < new_width; j++)
				ptr1[i][j]= ptr2[i][j];
			
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

	for(i=0; i<new_height; i++)
		delete[] ptr2[i];

	delete[] ptr1;
	delete[] ptr2;

	Invalidate();
}

void CImageProToolView::OnRotation()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.

	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();	// 불러온 문서변수 생성
	ASSERT_VALID(pDoc);

	int i, j;

	int width = pDoc->m_Width;	// 불러온 문서의 너비추출
	int height = pDoc->m_Height;// 불러온 문서의 높이추출

	BYTE* pData = pDoc->m_pDib->GetBitsAddress();	// 문서의 화소주소값을 pData에 저장

	RGBptr** ptr1 = new RGBptr*[height];	//pData의 값을 복사할 **ptr1 생성
	RGBptr** ptr2 = new RGBptr*[height];	// ptr1의 값을 조작할 때 저장할 **image생성

	/* **가 가르치는 *의 값 생성*/
	for (int i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		ptr2[i] = new RGBptr[width];
	}

	Seperate_RGB(pData, ptr1);	// pData의 값을 **ptr1에 저장

	int center_x = width / 2;
	int center_y = height / 2;

	double angle = 3.14159265/180*45;

	double x1, y1, x2, y2;

	int x, y;

	for(i=0; i<height; i++)
		for (j = 0; j < width; j++)
		{
			x1 = (j - center_x)*cos(angle);
			y1 = -1 * ((i - center_y)*sin(angle));
			x = (int)(x1 + y1 + center_x);

			x2 = (j - center_x)*sin(angle);
			y2 = (i - center_y)*cos(angle);
			y = (int)(x2 + y2 + center_y);

			if (y < 0 || y >= (height - 1) || x < 0 || x >= (width - 1))
				ptr2[i][j].r = ptr2[i][j].g = ptr2[i][j].b = 0;
			else
			{
				ptr2[i][j].r = (double)ptr1[y][x].r;
				ptr2[i][j].g = (double)ptr1[y][x].g;
				ptr2[i][j].b = (double)ptr1[y][x].b;
			}
		}


	SetRGBptr(pData, ptr2, width, height);

	for (i = 0; i < height; i++)
	{
		delete[] ptr1[i];
		delete[] ptr2[i];
	}

	delete[] ptr1;
	delete[] ptr2;

	Invalidate();
}

void CImageProToolView::OnBinarydilation()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.


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

	for(i=0; i<height; i++)
		for (j = 0; j < width; j++)
		{
			if (image[i][j] > 156)
				temp_image[i][j] = 255;
			else
				temp_image[i][j] = 0;
		}

	for(i=1; i<height-1; i++)
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


void CImageProToolView::OnBinaryerosion()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.


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
			if (image[i][j] > 156)
				temp_image[i][j] = 255;
			else
				temp_image[i][j] = 0;
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

void CImageProToolView::OnConnectedlabeling()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.

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

		for (i = 0; i <height; i++)
			for (j = 0; j<width; j++)
				image[i][j] = (BYTE)Saturation((ptr1[i][j].r + ptr1[i][j].g + ptr1[i][j].b) / 3);

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
				if (image[i][j] > 128) { Label[i][j] = 1; }
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
		int cnt = 1;
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


void CImageProToolView::Oninvariantmoment()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();
	ASSERT_VALID(pDoc);

	int width = pDoc->m_Width;
	int height = pDoc->m_Height;

	BYTE* pData = pDoc->m_pDib->GetBitsAddress();

	RGBptr** ptr1 = new RGBptr*[height];
	BYTE** image = new BYTE*[height];

	for (int i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		image[i] = new BYTE[width];
	}

	Seperate_RGB(pData, ptr1);

	int i, j;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			image[i][j] = (BYTE)Saturation((ptr1[i][j].r + ptr1[i][j].g + ptr1[i][j].b) / 3);
		}
	}

	//임의의 임계값으로 이진화 (전역 임계값을 적용해도 됨)
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (image[i][j] <= 0) { image[i][j] = 0; }
			else { image[i][j] = 255; }
		}
	}

	double mu00, mu11, mu20, mu02, mu30, mu03, mu21, mu12;
	double eta20, eta02, eta11, eta30, eta03, eta21, eta12;
	double *phi = new double[7];
	double count = 0., CenX = 0., CenY = 0.;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			if (image[i][j] > 0)
			{
				count++;
				CenX += j;
				CenY += i;
			}
		}
	}
	//영역의 무게 중심 구하기
	CenX /= count;
	CenY /= count;

	//중심 모멘트 구하기
	mu00 = mu11 = mu20 = mu02 = mu30 = mu03 = mu21 = mu12 = 0;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (image[i][j] > 0) //0보다 큰 영역에서만 적용 (레이블된 경우라면?)
			{
				mu00 += pow((double)(j - CenX), 0)*pow((double)(i - CenY), 0.);
				mu20 += pow((double)(j - CenX), 2.)*pow((double)(i - CenY), 0.);
				mu02 += pow((double)(j - CenX), 0.)*pow((double)(i - CenY), 2.);
				mu11 += pow((double)(j - CenX), 1.)*pow((double)(i - CenY), 1.);
				mu30 += pow((double)(j - CenX), 3.)*pow((double)(i - CenY), 0.);
				mu03 += pow((double)(j - CenX), 0.)*pow((double)(i - CenY), 3.);
				mu21 += pow((double)(j - CenX), 2.)*pow((double)(i - CenY), 1.);
				mu12 += pow((double)(j - CenX), 1.)*pow((double)(i - CenY), 2.);
			}
		}
	}

	//중심 모멘트 정규화 하기
	eta20 = mu20 / pow(mu00, (2. + 0.) / 2. + 1.);
	eta02 = mu02 / pow(mu00, (0. + 2.) / 2. + 1.);
	eta11 = mu11 / pow(mu00, (1. + 1.) / 2. + 1.);
	eta30 = mu30 / pow(mu00, (3. + 0.) / 2. + 1.);
	eta03 = mu03 / pow(mu00, (0. + 3.) / 2. + 1.);
	eta21 = mu21 / pow(mu00, (2. + 1.) / 2. + 1.);
	eta12 = mu12 / pow(mu00, (1. + 2.) / 2. + 1.);

	//불변 모멘트 구하기
	phi[0] = eta20 + eta02;
	phi[1] = (eta20 - eta02)*(eta20 - eta02) + 4 * eta11*eta11;
	phi[2] = (eta30 - 3 * eta12)*(eta30 - 3 * eta12) + (3 * eta21 - eta03)*(3 * eta21 - eta03);
	phi[3] = (eta30 + eta12)*(eta30 + eta12) + (eta21 + eta03)*(eta21 + eta03);
	phi[4] = (eta30 - 3 * eta12)*(eta30 + eta12)*((eta30 + eta12)*(eta30 + eta12) - 3 * (eta21 + eta03)*(eta21 + eta03))
		+ (3 * eta21 - eta03)*(eta21 + eta03)*(3 * (eta30 + eta12)*(eta30 + eta12) - (eta21 + eta03)*(eta21 + eta03));
	phi[5] = (eta20 - eta02)*((eta30 + eta12)*(eta30 + eta12) - (eta21 + eta03)*(eta21 + eta03))
		+ 4 * eta11*(eta30 + eta12)*(eta21 + eta03);
	phi[6] = (3 * eta21 - eta03)*(eta30 + eta12)*((eta30 + eta12)*(eta30 + eta12) - 3 * (eta21 + eta03)*(eta21 + eta03))
		+ (3 * eta12 - eta30)*(eta21 + eta03)*(3 * (eta30 + eta12)*(eta30 + eta12) - (eta21 + eta03)*(eta21 + eta03));

	CString str = _T("Invariant moments:n\n");
	for (int i = 0; i < 7; i++)
	{
		str.AppendFormat(_T("m[%d] = %10.10lf\n"), i, phi[i] * 1000);
	}

	AfxMessageBox(str);

	for (i = 0; i < height; i++) {
		delete[] ptr1[i];
		delete[] image[i];
	}
	delete[] ptr1;
	delete[] image;
	delete[] phi;
	Invalidate();
}


void CImageProToolView::OnKmeans()
{
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();
	ASSERT_VALID(pDoc);

	int width = pDoc->m_Width;
	int height = pDoc->m_Height;

	BYTE* pData = pDoc->m_pDib->GetBitsAddress();
	RGBptr** image = new RGBptr*[height];
	BYTE** grayImage = new BYTE*[height];

	for (int i = 0; i < height; i++)
	{
		image[i] = new RGBptr[width];
		grayImage[i] = new BYTE[width];
	}

	Seperate_RGB(pData, image);
	BYTE gray;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			gray = (BYTE)Saturation((image[i][j].r + image[i][j].g + image[i][j].b) / 3);
			image[i][j].r = image[i][j].g = image[i][j].b = gray;
			grayImage[i][j] = gray;
		}
	}
	
	int K = 5, T = 1;
	int*ME = new int[K];

	srand((unsigned)time(NULL));
	for (int i = 0; i < K; i++) ME[i] = rand() % (255);
	K_Mean_Clustering(grayImage, ME, K, T, width, height);
	
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			image[i][j].r = image[i][j].g = image[i][j].b = grayImage[i][j];
	//화면 출력을 위한 과정

	SetRGBptr(pData, image, width, height);

	for (int i = 0; i < height; i++)
	{
		delete[] image[i];
		delete[] grayImage[i];
	}
	delete[] image;
	delete[] grayImage;

	Invalidate();
}

void CImageProToolView::K_Mean_Clustering(BYTE** image, int *ME, int K, int T, int width, int height)
{
	float *C_mean = new float[K];
	float *distance = new float[K];
	int *Count = new int[K];
	int *Cluster = new int[K];

	int **label = new int*[height];
	for (int i = 0; i < height; i++)
	{
		label[i] = new int[width];
		memset(label[i], 0, sizeof(int)*width);
	}

	float min, temp_mean, Sum_mean;
	for (int k = 0; k < K; k++)
	{
		C_mean[k] = distance[k] = 0.0f;
		Count[k] = Cluster[k] = 0;
	}
	Sum_mean = min = temp_mean = 0.0f;

	for (int k = 0; k < K; k++)
	{
		C_mean[k] = (float)ME[k];
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			for (int k = 0; k < K; k++)
			{
				distance[k] = (float)fabs(C_mean[k] - (float)image[i][j]);
			}
			min = distance[0];
			int s = 0;

			for (int k = 1; k < K; k++)
			{
				if (distance[k] <= min) {
					min = distance[k];
					s = k;
				}
				Cluster[s] += image[i][j];
				label[i][j] = s;
				Count[s]++;
			}
		}
	}

	for (int k = 0; k < K; k++)
	{
		temp_mean = C_mean[k];
		if (Count[k] == 0) {
			C_mean[k] = 0.0;
		}
		else {
			C_mean[k] = (float)(Cluster[k] / Count[k]);
		}
		Sum_mean += (float)fabs(C_mean[k] - temp_mean);
		temp_mean = 0.0;
	}

	if (Sum_mean <= T) {
		for (int k = 0; k < K; k++)
		{
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					if (label[i][j] == k) {
						image[i][j] = (unsigned char)C_mean[k];
					}
				}
			}
		}
		return;
	}
	else {
		Sum_mean = 0.0;
		for (int k = 0; k < K; k++)
		{
			ME[k] = (int)C_mean[k];
		}
		K_Mean_Clustering(image, ME, K, T, width, height);
	}

	for (int i = 0; i < height; i++)
	{
		delete[] label[i];
	}
	delete[] label;
	delete[] C_mean;
	delete[] distance;
	delete[] Count;
	delete[] Cluster;
}