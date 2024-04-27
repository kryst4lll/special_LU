#include<iostream>
#include<vector>
#include <fstream>
#include <sstream>
#include<windows.h>
#include <xmmintrin.h> //SSE


using namespace std;


int N = 130; //矩阵行列数
vector<vector<float>> Elm_row;//被消元行 每个元素代表一行
vector<vector<float>> Elm_elm;//消元子 每个元素代表一行
vector<vector<float>> result;//被消元行结果

//返回首项
int get_first_nonzero_index(vector<float>& vec) {
    for (int i = vec.size() - 1; i >= 0; i--) {
        if (vec[i] != 0)
        {
            return i; // 返回第一个非零项的位置
        }

    }

    return -1; // 如果向量中都是零，则返回-1
}

void load()
{

    ifstream fileElement("消元子.txt"); // 打开文件

    string line;
    if (fileElement.is_open()) { // 确保文件成功打开
        while (getline(fileElement, line)){ // 逐行读取文件内容

            istringstream iss(line); // 使用字符串流按空格分隔每一行

            bool is_first = true;
            int number;
            int first;
            while (iss >> number) { // 从字符串流中读取整数
                
                if (is_first)
                {
                    first = number;
                    is_first = false;
                }
                Elm_elm[first][number] = 1;
            }
        }
        fileElement.close(); // 关闭文件
    }
    else {
        cerr << "Unable to open file" << endl; // 输出错误信息
    }

    ifstream fileCow("被消元行.txt"); // 打开文件
    if (fileCow.is_open()) { // 确保文件成功打开
        while (getline(fileCow, line)) { // 逐行读取文件内容
            istringstream iss(line); // 使用字符串流按空格分隔每一行
            bool is_first = true;
            int number;
            int first;
            vector<float> t(N, 0);
            while (iss >> number) { // 从字符串流中读取整数
                t[number] = 1;
            }
            Elm_row.push_back(t);
        }
        fileCow.close(); // 关闭文件
    }
    else {
        cerr << "Unable to open file" << endl; // 输出错误信息
    }
}



//异或操作
void Xor(vector<float>& row, vector<float>& elm) {

    for (int i = 0; i < N; i++) {
        if (row[i] == elm[i]) {
            row[i] = 0;
        }
        else {
            row[i] = 1;
        }
    }
}

void Minus() {
    while (Elm_row.size() != 0) {
        //被消元行的首项索引
        int x = get_first_nonzero_index(Elm_row[0]);
        if (x == -1) {
            Elm_row.erase(Elm_row.begin());
            continue;
        }
        //首项相同位置的消元子为全0
        if (get_first_nonzero_index(Elm_elm[x]) == -1)
        {
            Elm_elm[x] = Elm_row[0];
            result.push_back(Elm_row[0]);
            Elm_row.erase(Elm_row.begin());
        }
        else
        {
            Xor(Elm_row[0], Elm_elm[x]);
        }
    }
}

//SIMD异或操作
void SIMD_Xor(vector<float>& row, vector<float>& elm) {
    int i = 0;
    for (; i + 4 <= N; i += 4) {
        __m128 ROW = _mm_loadu_ps(&row[i]);
        __m128 ELM = _mm_loadu_ps(&elm[i]);
        __m128 result = _mm_xor_ps(ROW, ELM);
        _mm_storeu_ps(&row[i], result); // 将结果写回到row向量中
    }
    for (; i < N; i++) {
        if (row[i] == elm[i]) {
            row[i] = 0;
        }
        else {
            row[i] = 1;
        }
    }

}

void SIMD_Minus() {
    while (Elm_row.size() != 0) {
        //被消元行的首项索引
        int x = get_first_nonzero_index(Elm_row[0]);
        if (x == -1) {
            Elm_row.erase(Elm_row.begin());
            continue;
        }
        //首项相同位置的消元子为全0
        if (get_first_nonzero_index(Elm_elm[x]) == -1)
        {
            Elm_elm[x] = Elm_row[0];
            result.push_back(Elm_row[0]);
            Elm_row.erase(Elm_row.begin());
        }
        else
        {
            SIMD_Xor(Elm_row[0], Elm_elm[x]);
        }
    }
}

//打印并清空结果
void Print() {
    for (int i = 0; i < result.size(); i++) {
        for (int j = result[0].size() - 1; j >= 0; j--) {
            if (result[i][j] == 1)
            {
                cout << j << " ";
            }
        }
        cout << endl;
    }
    result.clear();
}

int main() {
    cout << "输入矩阵行列数:";
    cin >> N;

    //特殊高斯消去法
    //初始化
    for (int i = 0; i < N; i++) {
        vector<float> t;
        for (int j = 0; j < N; j++) {
            float x = 0;
            t.push_back(x);
        }
        Elm_elm.push_back(t);
    }
    load();

    LARGE_INTEGER frequency;        // 声明一个LARGE_INTEGER类型的变量来存储频率
    LARGE_INTEGER start, end;       // 声明两个LARGE_INTEGER类型的变量来存储开始和结束的计数值
    double elapsedTime;             // 声明一个double类型的变量来存储经过的时间
    // 获取计时器的频率
    QueryPerformanceFrequency(&frequency);
    // 记录开始时间
    QueryPerformanceCounter(&start);
    Minus();
    // 记录结束时间
    QueryPerformanceCounter(&end);
    // 计算经过的时间
    elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    cout << "特殊高斯消去法时间: " << elapsedTime << " ms." << std::endl;
    //打印结果
    //Print();


    //SIMD特殊高斯消去法
    Elm_elm.clear();
    Elm_row.clear();
    //初始化
    for (int i = 0; i < N; i++) {
        vector<float> t;
        for (int j = 0; j < N; j++) {
            float x = 0;
            t.push_back(x);
        }
        Elm_elm.push_back(t);
    }
    load();

    // 获取计时器的频率
    QueryPerformanceFrequency(&frequency);
    // 记录开始时间
    QueryPerformanceCounter(&start);
    SIMD_Minus();
    // 记录结束时间
    QueryPerformanceCounter(&end);
    // 计算经过的时间
    elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    cout << "特殊高斯消去法时间: " << elapsedTime << " ms." << std::endl;

    //Print();


    return 0;
}