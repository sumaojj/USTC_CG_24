#include "comp_target_image.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <cmath>
#include <iostream>

namespace USTC_CG
{
using namespace Eigen;
using uchar = unsigned char;

CompTargetImage::CompTargetImage(
    const std::string& label,
    const std::string& filename)
    : ImageEditor(label, filename)
{
    if (data_)
        back_up_ = std::make_shared<Image>(*data_);
}

void CompTargetImage::draw()
{
    // Draw the image
    ImageEditor::draw();
    // Invisible button for interactions
    ImGui::SetCursorScreenPos(position_);
    ImGui::InvisibleButton(
        label_.c_str(),
        ImVec2(
            static_cast<float>(image_width_),
            static_cast<float>(image_height_)),
        ImGuiButtonFlags_MouseButtonLeft);
    bool is_hovered_ = ImGui::IsItemHovered();
    // When the mouse is clicked or moving, we would adapt clone function to
    // copy the selected region to the target.
    ImGuiIO& io = ImGui::GetIO();
    if (is_hovered_ && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
    {
        edit_status_ = true;
        mouse_position_ =
            ImVec2(io.MousePos.x - position_.x, io.MousePos.y - position_.y);
        clone();
    }
    if (edit_status_)
    {
        mouse_position_ =
            ImVec2(io.MousePos.x - position_.x, io.MousePos.y - position_.y);
        if (flag_realtime_updating)
            clone();
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left))
        {
            edit_status_ = false;
        }
    }
}

void CompTargetImage::set_source(std::shared_ptr<CompSourceImage> source)
{
    source_image_ = source;
}

void CompTargetImage::set_realtime(bool flag)
{
    flag_realtime_updating = flag;
}

void CompTargetImage::restore()
{
    *data_ = *back_up_;
    update();
}

void CompTargetImage::set_paste()
{
    clone_type_ = kPaste;
}

void CompTargetImage::set_seamless()
{
    clone_type_ = kSeamless;
}

void CompTargetImage::clone()
{
    // The implementation of different types of cloning
    // HW3_TODO: In this function, you should at least implement the "seamless"
    // cloning labeled by `clone_type_ ==kSeamless`.
    //
    // The realtime updating (update when the mouse is moving) is only available
    // when the checkboard is selected. It is required to improve the efficiency
    // of your seamless cloning to achieve realtime editing. (Use decomposition
    // of sparse matrix before solve the linear system)
    if (data_ == nullptr || source_image_ == nullptr ||
        source_image_->get_region() == nullptr)
        return;
    std::shared_ptr<Image> mask = source_image_->get_region();

    switch (clone_type_)
    {
        case USTC_CG::CompTargetImage::kDefault: break;
        case USTC_CG::CompTargetImage::kPaste:
        {
            restore();

            for (int i = 0; i < mask->width(); ++i)
            {
                for (int j = 0; j < mask->height(); ++j)
                {
                    int tar_x =
                        static_cast<int>(mouse_position_.x) + i -
                        static_cast<int>(source_image_->get_position().x);
                    int tar_y =
                        static_cast<int>(mouse_position_.y) + j -
                        static_cast<int>(source_image_->get_position().y);
                    if (0 <= tar_x && tar_x < image_width_ && 0 <= tar_y &&
                        tar_y < image_height_ && mask->get_pixel(i, j)[0] > 0)
                    {
                        data_->set_pixel(
                            tar_x,
                            tar_y,
                            source_image_->get_data()->get_pixel(i, j));
                    }
                }
            }
            break;
        }
        case USTC_CG::CompTargetImage::kSeamless:
        {
            // You should delete this block and implement your own seamless
            // cloning. For each pixel in the selected region, calculate the
            // final RGB color by solving Poisson Equations.
            restore();
            // get the size of the selected region
            int width = 0;
            int height = 0;
            for (int i = static_cast<int>(source_image_->get_position().y);
                 i < mask->height();
                 i++)
            {
                if (mask->get_pixel(
                        static_cast<int>(source_image_->get_position().x),
                        i)[0] > 0)
                {
                    height++;
                }
            }

            for (int j = static_cast<int>(source_image_->get_position().x);
                 j < mask->width();
                 j++)
            {
                if (mask->get_pixel(
                        j,
                        static_cast<int>(source_image_->get_position().y))[0] >
                    0)
                {
                    width++;
                }
            }

            width = width - 2;
            height = height - 2;
            // std::cout << "width" << width << "height" << height << std::endl;

            // get inside Laplacian value of the slected image
            int x_0 = static_cast<int>(source_image_->get_position().x);
            int y_0 = static_cast<int>(source_image_->get_position().y);
            MatrixXd b(width * height, 3);
            for (int k = 0; k < 3; k++)
            {
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        if (mask->get_pixel(x_0 + i + 1, y_0 + j + 1)[0] > 0)
                        {
                            b(i * height + j, k) =
                                (4 * source_image_->get_data()->get_pixel(
                                         x_0 + i + 1, y_0 + j + 1)[k] -
                                 source_image_->get_data()->get_pixel(
                                     x_0 + i, y_0 + j + 1)[k] -
                                 source_image_->get_data()->get_pixel(
                                     x_0 + i + 2, y_0 + j + 1)[k] -
                                 source_image_->get_data()->get_pixel(
                                     x_0 + i + 1, y_0 + j)[k] -
                                 source_image_->get_data()->get_pixel(
                                     x_0 + i + 1, y_0 + j + 2)[k]);
                        }
                    }
                }
            }
            // std::cout << "b" << b << std::endl;

            // if

            // modify the Matrix b to match the boundary condition
            for (int k = 0; k < 3; k++)
            {
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        if (i == 0)
                        {
                            b(i * height + j, k) +=
                                data_->get_pixel(x_0 + i, y_0 + j + 1)[k];
                        }
                        if (i == width - 1)
                        {
                            b(i * height + j, k) +=
                                data_->get_pixel(x_0 + i + 2, y_0 + j + 1)[k];
                        }
                        if (j == 0)
                        {
                            b(i * height + j, k) +=
                                data_->get_pixel(x_0 + i + 1, y_0 + j)[k];
                        }
                        if (j == height - 1)
                        {
                            b(i * height + j, k) +=
                                data_->get_pixel(x_0 + i + 1, y_0 + j + 2)[k];
                        }
                    }
                }
            }
            // std::cout << "b" << b << std::endl;

            // entry Matrix
            SparseMatrix<double> A(width * height, width * height);
            std::vector<Triplet<double>> triplets;
            A.setZero();
            for (int i = 0; i < width * height; i++)
            {
                triplets.push_back(Triplet<double>(i, i, 4));
            }
            for (int i = 0; i < width * height - 1; i++)
            {
                triplets.push_back(Triplet<double>(i, i + 1, -1));
                triplets.push_back(Triplet<double>(i + 1, i, -1));
            }
            for (int i = 0; i < width * height - height; i++)
            {
                triplets.push_back(Triplet<double>(i, i + height, -1));
                triplets.push_back(Triplet<double>(i + height, i, -1));
            }
            A.setFromTriplets(triplets.begin(), triplets.end());

            // std::cout << "A" << A << std::endl;

            // solve the sparse linear system
            SparseLU<SparseMatrix<double>> solver;
            solver.compute(A);
            if (solver.info() != Success)
            {
                std::cout << "Error in solving the linear system" << std::endl;
                return;
            }
            // MatrixXd x(width * height, 3);
            // for (int i = 0; i < 3; i++)
            // {
            //     x.col(i) = solver.solve(b.col(i));
            //     if (solver.info() != Success)
            //     {
            //         std::cout << "Error in solving the linear system2"
            //                   << std::endl;
            //         return;
            //     }
            // }
            MatrixXd x = solver.solve(b);
            // std::cout << "x" << x << std::endl;

            // if entry>255, entry=255
            for (int i = 0; i < width * height; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    if (x(i, j) > 255)
                    {
                        x(i, j) = 255;
                    }
                }
            }

            // modify the target image

            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    int tar_x =
                        static_cast<int>(mouse_position_.x) + i -
                        static_cast<int>(source_image_->get_position().x);
                    int tar_y =
                        static_cast<int>(mouse_position_.y) + j -
                        static_cast<int>(source_image_->get_position().y);
                    if (0 <= tar_x && tar_x < image_width_ && 0 <= tar_y &&
                        tar_y < image_height_ &&
                        mask->get_pixel(x_0 + i + 1, y_0 + j + 1)[0] > 0)
                    {
                        data_->set_pixel(
                            tar_x,
                            tar_y,
                            { static_cast<uchar>(x(i * height + j, 0)),
                              static_cast<uchar>(x(i * height + j, 1)),
                              static_cast<uchar>(x(i * height + j, 2)) });
                    }
                }
            }

            break;
        }
        default: break;
    }

    update();
}

}  // namespace USTC_CG