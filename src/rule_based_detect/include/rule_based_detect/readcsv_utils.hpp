#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// // CSV 파일 경로와 읽고자 하는 열 인덱스를 정의합니다.
// const std::string CSV_FILE_PATH = "map_l_xy.csv";
// const int COLUMN_INDEX = 1; // 0부터 시작하는 열 인덱스

// // CSV 파일에서 특정 열의 데이터를 읽어와 벡터로 반환하는 함수
std::vector<double> readColumnData(const std::string& filePath, int columnIndex) {
    std::vector<double> columnData;
    std::ifstream file(filePath);
    std::string line;

    // 파일 경로와 파일 존재 여부 확인을 위한 디버깅 출력
    // std::cout << "파일 경로: " << filePath << std::endl;
    // if (access(filePath.c_str(), F_OK) != -1) {
    //     std::cout << "파일이 존재합니다." << std::endl;
    // } else {
    //     std::cerr << "파일이 존재하지 않습니다: " << filePath << std::endl;
    //     return columnData;
    // }

    // if (!file.is_open()) {
    //     std::cerr << "파일을 열 수 없습니다: " << filePath << std::endl;
    //     return columnData;
    // }

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> row;
        
        while (std::getline(lineStream, cell, ',')) {
            row.push_back(cell);
        }
        
        if (row.size() > columnIndex) {
            try {
                double value = std::stod(row[columnIndex]);
                columnData.push_back(value);
            } catch (const std::invalid_argument& e) {
                std::cerr << "잘못된 숫자 형식: " << e.what() << std::endl;
            } catch (const std::out_of_range& e) {
                std::cerr << "숫자 범위 초과: " << e.what() << std::endl;
            }
        } else {
            std::cerr << "열 인덱스가 범위를 벗어났습니다: " << columnIndex << std::endl;
        }
    }

    file.close();
    return columnData;
}

// int main() {
//     std::vector<double> data = readColumnData(CSV_FILE_PATH, COLUMN_INDEX);

//     std::cout << "읽은 데이터: " << std::endl;
//     for (const auto& item : data) {
//         std::cout << item << std::endl;
//     }

//     return 0;
// }