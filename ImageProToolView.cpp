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


double pphi[7] = {0.0f,};	// Invariant Moment 값 저장을 위한 전역변수 선언

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
	ON_COMMAND(ID_CHAP_MOMENT, &CImageProToolView::OnIMoment)
	ON_COMMAND(ID_CHAP_THRESHOLD, &CImageProToolView::OnThreshold)
	ON_COMMAND(ID_CHAP_CONNECTEDLABELING, &CImageProToolView::OnConnectedlabeling)
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
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();	//파일로드
	ASSERT_VALID(pDoc);	// pDOc의 객체 유효성 확인

	int width = pDoc->m_Width;	// 문서의 너비로드
	int height = pDoc->m_Height;	// 문서의 높이 로드
	int i, j;

	BYTE* pData = pDoc->m_pDib->GetBitsAddress();	// 문서의 첫 주소값 로드
	RGBptr** ptr1 = new RGBptr*[height];	// R, G, B 세 가지 채널을 위한 RGBptr 타입선언


	for (i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
	}

	Seperate_RGB(pData, ptr1);	// pData을 R, G, B 세 채널로 ptr1에 저장

	int tempR[256], tempG[256], tempB[256], tempgray[256];	// 누적히스토그램을 위한 변수 선언
	float histoR[256] = { 0.f, }, histoG[256] = { 0.f, }, histoB[256] = { 0.f, }, histogray[256] = { 0.f, };	// 정규화 히스토그램을 위한 변수 선언
	/*변수 초기화*/
	memset(tempgray, 0, sizeof(int) * 256);
	memset(tempR, 0, sizeof(int) * 256);
	memset(tempG, 0, sizeof(int) * 256);
	memset(tempB, 0, sizeof(int) * 256);
	
	/*누적 히스토그램 계산*/
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



	Histogram dlg; // Histogram 객체의 Dialog 생성
	dlg.SetImage(histogray, histoR, histoG, histoB);
	dlg.DoModal();
}



void CImageProToolView::OnIMoment()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CImageProToolView::OnThreshold();	// Thresholding 실행

	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();	// 문서 로드
	ASSERT_VALID(pDoc);	// pDoc의 객체 유효성확인
	int width = pDoc->m_Width;	// 문서의 너비로드
	int height = pDoc->m_Height;	// 문서의 높이로드
	BYTE* pData = pDoc->m_pDib->GetBitsAddress();	//문서의 주소 로드
	/*영상조작을 위한 추가적인 변수선언*/
	RGBptr** ptr1 = new RGBptr*[height];
	BYTE** image = new BYTE*[height];

	double *phi = new double[7];	//Invarint Moment 저장을 위한 변수 선언

	for (int i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
		image[i] = new BYTE[width];
	}

	Seperate_RGB(pData, ptr1);	// pData의 R,G,B 채널에대해서 ptr1에 저장


	int i, j;
	/* Gray값 저장*/
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			image[i][j] =
				(BYTE)Saturation((ptr1[i][j].r + ptr1[i][j].g + ptr1[i][j].b) / 3);
		}
	}
	/*임의의 임계값으로 이진화*/
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (image[i][j] <= 0) { image[i][j] = 0; }
			else { image[i][j] = 255; }
		}
	}
	
	/*Moment 값들을 저장하기위한 변수들 선언*/
	double mu00, mu11, mu20, mu02, mu30, mu03, mu21, mu12;
	double eta20, eta02, eta11, eta30, eta03, eta21, eta12;

	/*영역무게의 중심계산*/
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

	
	
	for (i = 0; i < 7; i++)
	{
		pphi[i] = phi[i];	//Query 영상의 Invariant Moment 값을 전역변수에 저장
	}

	/*Invariant Moment 값 출력*/
	CString str = _T("Invariant moments:n\n");
	for (int i = 0; i < 7; i++)
	{
		str.AppendFormat(_T("m[%d] = %10.10lf\n"), i, pphi[i] * 1000);
	}

	AfxMessageBox(str);

	/*소멸자 호출*/
	for (i = 0; i < height; i++) {
		delete[] ptr1[i];
		delete[] image[i];
	}
	delete[] ptr1;
	delete[] image;
	delete[] phi;
	Invalidate();
}


void CImageProToolView::OnThreshold()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();	// 문서 로드
	ASSERT_VALID(pDoc);	// pDoc의 객체 유효성확인

	int width = pDoc->m_Width;	// 문서의 너비로드
	int height = pDoc->m_Height;	// 문서의 높이로드
	BYTE* pData = pDoc->m_pDib->GetBitsAddress();	//문서의 주소 로드
	int i, j;

	/*영상 조작을 위한 RGBptr 변수 선언 및 초기화*/
	RGBptr** ptr1 = new RGBptr*[height];

	for (int i = 0; i < height; i++)
	{
		ptr1[i] = new RGBptr[width];
	}

	Seperate_RGB(pData, ptr1);
	int T1 = 120;	// 임계값 지정
	int* Y = new int[height*width];

	memset(Y, 0, height*width);	// 초기화

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

	/*소멸자 호출*/
	for (int i = 0; i < height; i++)
	{
		delete[] ptr1[i];
	}
	delete[] ptr1;
	delete[] Y;
	Invalidate();
}


void CImageProToolView::OnConnectedlabeling()
{
	CImageProToolView::OnThreshold();	//Threshold 실행

	CImageProToolDoc* pDoc = (CImageProToolDoc*)GetDocument();	// 문서로드
	ASSERT_VALID(pDoc);	// pDoc의 객체 유효성검사

	int width = pDoc->m_Width;	// pDoc의 너비저장
	int height = pDoc->m_Height;	// pDoc의 높이 저장

	BYTE* pData = pDoc->m_pDib->GetBitsAddress();	// 문서의 주소로드
	/*영상 조작을 위한 변수선언*/
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
			image[i][j] = (BYTE)Saturation((ptr1[i][j].r + ptr1[i][j].g + ptr1[i][j].b) / 3);	// Gray값으로 초기화

	int minRegionCount = 10;	// 지역에 지정된 픽셀보드 수가 적은경우 제거

	/*labeling 변수 선언*/
	int** Label = new int*[height];
	for (i = 0; i < height; i++)
	{
		Label[i] = new int[width];
		memset(Label[i], 0, sizeof(int)*width);
	}

	/*등가 테이블 및 필요요소 생성 및 초기화*/
	int num, left, top, k;
	int *r, *area;
	r = new int[height*width];
	area = new int[width*height];
	memset(r, 0, sizeof(int)*height*width);
	memset(area, 0, sizeof(int)*height*width);
	
	/*Object 구분을 위한 이진화 실행*/
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			if (image[i][j] > 128) { Label[i][j] = 1; }
			else { Label[i][j] = -1; }

	/*영상의 경계부분은 제외*/
	for (j = 0; j < width; j++) {
		Label[0][j] = -1;
		Label[height - 1][j] = -1;
	}
	for (i = 0; i < height; i++) {
		Label[i][0] = -1;
		Label[i][width - 1] = -1;
	}

	num = 0; // 지정할 레이블 number 초기화
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			if (i > 0 && j > 0) {	// pixel이 0이상일 때 검사
				if (Label[i][j] >= 1) {
					left = Label[i][j - 1]; // 좌측 픽셀 저장 
					top = Label[i - 1][j];  // 상단 픽셀 저장
					if (left == -1 && top != -1) {  // 상단 픽셀만 존재한다면
							// 상단의 레이블 number로 저장
						Label[i][j] = r[top];
					}
					// 좌측 픽셀만 존재할 경우
					else if (left != -1 && top == -1) {
						// 좌측의 레이블 number로 저장
						Label[i][j] = r[left];
					}
					// 좌측, 상단 픽셀 모두 존재하지 않을 경우
					else if (left == -1 && top == -1) {
						num++;             // label number을 1 증가
							   // 등가테이블에도 저장
						r[num] = num;
						// lable number로 지정   
						Label[i][j] = r[num];
					}
					//둘다 값이 있을경우
					//1) 같은 값이면 좌측의 값으로
					//2) 아니면 작은 쪽의 값으로
					// 변경
					else if (left != -1 && top != -1) {
						if (r[left] == r[top]) {
							Label[i][j] = r[left];
						}
						else if (r[left] > r[top]) {
							Label[i][j] = r[top];
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
	/*Label 재정리*/
	for (k = 1; k <= num; k++) { 
		   //k와 해당 레이블링 번호가 같지 않다면 새롭게 설정 
		if (k != r[k]) r[k] = r[r[k]];
		area[k] = 0;
	}

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++) {
			if (Label[i][j] > 0) {
				//새로운 레이블값으로 다시 설정
				Label[i][j] = r[Label[i][j]];
				// 각 레이블 영역의 개수 구하기
				area[Label[i][j]]++;
			}
		}
	int cnt = 1;
	for (k = 1; k <= num; k++) {
		//임계값 이하인 영역은 제거하고 레이블 번호 다시부여
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
				//레이블 번호 다시부여
				Label[i][j] = r[Label[i][j]];
		}
	//Invariant Moment 계산을 위한 변수선언
	double count, CenX, CenY;
	double* temp_phi = new double[7];
	count = CenX = CenY = 0.;

	int shark_label[10] = { 0, };	// 찾는 Object의 Label을 저장할 배열 선언
	int shark_count = 0;	// Object의 개수

	for (int counts = 1; counts <= cnt; counts++)
	{
		for (i = 0; i < height; i++)
			for (j = 0; j < width; j++) {
				if (Label[i][j] == counts)
				{
					count++;
					CenX += j;
					CenY += i;
				}
			}
		// 중심값 계산
		CenX /= count;
		CenY /= count;
		temp_phi = CImageProToolView::invariantMomento(Label, width, counts, height, count, CenX, CenY);	// Invariant Moment값 계산
		
		if (fabs((pphi[1]*1000) - (temp_phi[1]*1000))<24 && fabs((pphi[2] * 1000) - (temp_phi[2] * 1000))<6 ){	// 임계값의 범위내에 들어오는지 검사
			
			shark_label[shark_count++] = counts;	//발견할 경우 해당 labeling number 저장
			CString str;
			str.Format(_T("발견"));
			AfxMessageBox(str);

			/*원하는 Object를 구분하기위해 '-5'값으로 설정*/
			for (i = 0; i < height; i++)
				for (j = 0; j < width; j++) {
					if(Label[i][j] == counts)
						Label[i][j] = -5;
				}
		}
		count = CenX = CenY = 0.;	// 값 초기화
	}

	/*원하는 Object의 값만 나타나도록 설정*/
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			if (Label[i][j] == -5) {
				Label[i][j] = 255;
			}
			else Label[i][j] = 0;
		}
	}

	CString stre = _T("Invariant s:n\n");
	stre.Format(_T("%d"), shark_count);
	AfxMessageBox(stre);

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			ptr1[i][j].r = Label[i][j]*10;
			ptr1[i][j].g = Label[i][j] * 10;
			ptr1[i][j].b = Label[i][j] * 10;
		}

	SetRGBptr(pData, ptr1, width, height);

	/*소멸자 호출*/
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

/* Invariant Moment 계산을 위한 함수생성*/
double* CImageProToolView::invariantMomento(int **Label, int width, int labelNumber, int height, double count, double CenX, double CenY)
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	int i, j;
	// 영상조작을 위한 변수 생성 및 초기화
	int **image = new int*[height];
	for (i = 0; i < height; i++)
		image[i] = new int[width];

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			image[i][j] = Label[i][j];

	// 모멘토 계산을 위한 변수들 선언
	double mu00, mu11, mu20, mu02, mu30, mu03, mu21, mu12;
	double eta20, eta02, eta11, eta30, eta03, eta21, eta12;
	double *phi = new double[7];

	//중심 모멘트 구하기
	mu00 = mu11 = mu20 = mu02 = mu30 = mu03 = mu21 = mu12 = 0;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (image[i][j] == labelNumber) // labelNumber 영역에 대해서만 계산
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

	return phi;	// Invariant Moment 반환
}
