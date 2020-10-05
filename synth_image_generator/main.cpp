//g++-7 main.cpp -o test -lGL -lGLU -lGLEW `pkg-config --static --libs glfw3` -lSOIL -lassimp -lfreeimage -lstdc++fs -std=c++17
//./test

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <experimental/filesystem>

// GLEW
#define GLEW_STATIC
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>

// GLM Mathematics
#include <glm/gtc/type_ptr.hpp>

#include <FreeImage.h>
#include <random>


// Other includes
#include "Shader.h"
#include "Model.h"

namespace fs = std::experimental::filesystem;

// Window dimensions
const GLuint WIDTH = 1024, HEIGHT = 576;
int SCREEN_WIDTH, SCREEN_HEIGHT;
int num_piezas = 3;
int start_back = 8;
int end_back = 100;
int num_backgrounds = end_back - start_back + 1;
int num_transf = 11;
std::string dir_images = "new_images";
std::string dir_backgrounds = "new_backgrounds_v2";
std::string dir_models = "models";

glm::vec3 position_camera(0.0f, 0.0f, 4.0f);

// The MAIN function, from here we start the application and run the game loop
int main(int argc, char** argv)
{ 
    std::random_device dev;
    std::mt19937 rng(dev());

    // Init GLFW
    glfwInit( );
    // Set all the required options for GLFW
    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );
    glfwWindowHint( GLFW_RESIZABLE, GL_FALSE );
    
    // Create a GLFWwindow object that we can use for GLFW's functions
    GLFWwindow* window = glfwCreateWindow( WIDTH, HEIGHT, "Programa Fondecyt", nullptr, nullptr );
    
    if ( nullptr == window )
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate( );
        
        return EXIT_FAILURE;
    }
    
    glfwMakeContextCurrent( window );
    
    glfwGetFramebufferSize( window, &SCREEN_WIDTH, &SCREEN_HEIGHT );
      
    // GLFW Options
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    // Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions
    glewExperimental = GL_TRUE;
    // Initialize GLEW to setup the OpenGL Function pointers
    if ( GLEW_OK != glewInit( ) )
    {
        std::cout << "Failed to initialize GLEW" << std::endl;
        return EXIT_FAILURE;
    } 

    // Define the viewport dimensions
    glViewport( 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT );
    
    // OpenGL options
    glEnable( GL_DEPTH_TEST ); 

	std::string vertex_shader_path;
	std::string fragment_shader_path;

	vertex_shader_path = 
	"E://Alonso//Alonso//src//image_generator//modelLoading.vs";
	fragment_shader_path = 
	"E://Alonso//Alonso//src//image_generator//modelLoading.frag";


    //Shader shader("modelLoading.vs", "modelLoading.frag");
	Shader shader(vertex_shader_path.c_str(), fragment_shader_path.c_str());
    
	std::vector<std::vector<Model*>> models_backgrounds(num_piezas, std::vector<Model*>(num_backgrounds));

	std::string base_path =
	"E://Alonso//Alonso//src//image_generator";

	for (int j = 0; j < num_piezas; ++j)
	{
		for (int i = start_back; i <= end_back; ++i)
		{
			//std::cout << path_to_backgrounds[i] << "\n";
			std::string name_model_back =
				base_path + "//" +
				//"./" + 
				dir_backgrounds + "/background" + std::to_string(i) + ".obj";
			models_backgrounds[j][i-start_back] = new Model((GLchar*)name_model_back.c_str());
		}
	}

	glm::mat4 model_background(1.0f);
	model_background = glm::scale(model_background, glm::vec3(8.0f, 8.0f, 1.0f));
	//model_background = glm::rotate(model_background, glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	model_background = glm::translate(model_background, glm::vec3(0.0f, 0.0f, -3.0f));

    glm::mat4 projection = glm::perspective(45.0f, (GLfloat)SCREEN_WIDTH / (GLfloat)SCREEN_HEIGHT, 0.1f, 100.0f);
    glm::mat4 view = glm::lookAt(position_camera, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 model(1.0f);

	std::vector<std::unordered_map<std::string, std::vector<float>>> params_transf(num_piezas);
	std::unordered_map<std::string, std::vector<float>> params_pieza0 =
	{
		{"x_rot", {0.0f}},
		{"y_rot", {180.0f}},
		{"z_rot", {0.0f}},
		{"dist_x_rot", {0.0f, 360.0f}},
		{"dist_y_rot", {-45.0f, 45.0f}},
		{"dist_z_rot", {-25.0f, 25.0f}},
		{"dist_scale", {0.004f, 0.009f}},
		{"dist_x_transl", {-150.0f, 150.0f}},
		{"dist_y_transl", {-150.0f, 150.0f}}
	};
	std::unordered_map<std::string, std::vector<float>> params_pieza1 =
	{
		{"x_rot", {0.0f}},
		{"y_rot", {0.0f}},
		{"z_rot", {0.0f}},
		{"dist_x_rot", {0.0f, 360.0f}},
		{"dist_y_rot", {0.0f, 360.0f}},
		{"dist_z_rot", {0.0f, 360.0f}},
		{"dist_scale", {0.005f, 0.013f}},
		{"dist_x_transl", {-100.0f, 100.0f}},
		{"dist_y_transl", {-100.0f, 100.0f}}
	};
	std::unordered_map<std::string, std::vector<float>> params_pieza2 =
	{
		{"x_rot", {90.0f}},
		{"y_rot", {180.0f}},
		{"z_rot", {0.0f}},
		{"dist_x_rot", {-25.0f, 25.0f}},
		{"dist_y_rot", {-45.0f, 45.0f}},
		{"dist_z_rot", {-25.0f, 25.0f}},
		{"dist_scale", {0.005f, 0.010f}},
		{"dist_x_transl", {-150.0f, 150.0f}},
		{"dist_y_transl", {-150.0f, 150.0f}}
	};
	params_transf[0] = params_pieza0;
	params_transf[1] = params_pieza1;
	params_transf[2] = params_pieza2;

    //fs::create_directory("./" + dir_images)
	fs::create_directory(  base_path + "//" + dir_images);

    for (int idx_model = 0; idx_model < num_piezas; ++idx_model)
    {
        //fs::create_directory("./" + dir_images + "/pieza" + std::to_string(idx_model));
		fs::create_directory( 
			base_path + "//" + dir_images + "/pieza" + std::to_string(idx_model));


        std::string name_model_pieza = 
			base_path + "//" +
			//"./" + 
			dir_models + "/pieza" + std::to_string(idx_model) + ".obj";
        
		
		Model ourModel((GLchar *) name_model_pieza.c_str());
        
        for (int cont_transf = 0; cont_transf < num_transf; ++cont_transf)
        {
            for (int idx_background = 0; idx_background < num_backgrounds; ++idx_background)
            {                                     
                // Pass the matrices to the shader
                shader.Use();
                
                GLint LightID = glGetUniformLocation(shader.Program, "LightPosition_worldspace");
                GLint modelLoc = glGetUniformLocation(shader.Program, "model");
                GLint viewLoc = glGetUniformLocation(shader.Program, "view");
                GLint projLoc = glGetUniformLocation(shader.Program, "projection");

                glm::vec3 lightPos = glm::vec3(4,4,4);
                glUniform3f(LightID, lightPos.x, lightPos.y, lightPos.z);
                glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
                glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

				std::uniform_real_distribution<float> dist_x_rot(params_transf[idx_model]["dist_x_rot"][0], params_transf[idx_model]["dist_x_rot"][1]);
				float x_rotation = dist_x_rot(rng);
				std::uniform_real_distribution<float> dist_y_rot(params_transf[idx_model]["dist_y_rot"][0], params_transf[idx_model]["dist_y_rot"][1]);
				float y_rotation = dist_y_rot(rng);
				std::uniform_real_distribution<float> dist_z_rot(params_transf[idx_model]["dist_z_rot"][0], params_transf[idx_model]["dist_z_rot"][1]);
				float z_rotation = dist_z_rot(rng);

				std::uniform_real_distribution<float> dist_scale(params_transf[idx_model]["dist_scale"][0], params_transf[idx_model]["dist_scale"][1]);
				float scale_model = dist_scale(rng);

				std::uniform_real_distribution<float> dist_x_transl(params_transf[idx_model]["dist_x_transl"][0], params_transf[idx_model]["dist_x_transl"][1]);
				std::uniform_real_distribution<float> dist_y_transl(params_transf[idx_model]["dist_y_transl"][0], params_transf[idx_model]["dist_y_transl"][1]);
				float x_translation = dist_x_transl(rng);
				float y_translation = dist_y_transl(rng);

				model = glm::mat4(1.0f);
				model = glm::scale(model, glm::vec3(scale_model, scale_model, scale_model));
				model = glm::rotate(model, glm::radians(params_transf[idx_model]["x_rot"][0] + x_rotation), glm::vec3(1.0f, 0.0f, 0.0f));
				model = glm::rotate(model, glm::radians(params_transf[idx_model]["y_rot"][0] + y_rotation), glm::vec3(0.0f, 1.0f, 0.0f));
				model = glm::rotate(model, glm::radians(params_transf[idx_model]["z_rot"][0] + z_rotation), glm::vec3(0.0f, 0.0f, 1.0f));
				model = glm::translate(model, glm::vec3(x_translation, y_translation, 1.0f));

                // Poll events
                glfwPollEvents();
                
                // Clear the colorbuffer
                
                glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                std::string name_image;

                glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
                ourModel.Draw(shader);

                name_image = 
					//"./" + 
					base_path + "//" +
 					dir_images + "/pieza" + std::to_string(idx_model) + "/img" + std::to_string(cont_transf) + "_background" + std::to_string(idx_background) + ".png";
                
                //to take screnshots during the movement
                std::vector<unsigned char> pixels(4 * WIDTH * HEIGHT);
                glReadPixels(0, 0, WIDTH, HEIGHT, GL_BGRA, GL_UNSIGNED_BYTE, &pixels[0]);

                FIBITMAP* image = FreeImage_ConvertFromRawBits(&pixels[0], WIDTH, HEIGHT, 4 * WIDTH, 32, 0x0000FF, 0xFF0000, 0x00FF00, false);
                FreeImage_Save(FIF_PNG, image, name_image.c_str(), 0);
                FreeImage_Unload(image);
     

                // Swap the screen buffers
                glfwSwapBuffers(window);

                glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
                ourModel.Draw(shader);
                
                glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model_background));
                models_backgrounds[idx_model][idx_background]->Draw(shader);

                name_image = 
					//"./" +
					base_path + "//" +
					dir_images + "/pieza" + std::to_string(idx_model) + "/img" + to_string(cont_transf) + "_background" + to_string(idx_background) + "_w.png";      
                
                //to take screnshots during the movement
                std::vector<unsigned char> pixels2(4 * WIDTH * HEIGHT);
                glReadPixels(0, 0, WIDTH, HEIGHT, GL_BGRA, GL_UNSIGNED_BYTE, &pixels2[0]);

                FIBITMAP* image2 = FreeImage_ConvertFromRawBits(&pixels2[0], WIDTH, HEIGHT, 4 * WIDTH, 32, 0x0000FF, 0xFF0000, 0x00FF00, false);
                FreeImage_Save(FIF_PNG, image2, name_image.c_str(), 0);
                FreeImage_Unload(image2);

                // Swap the screen buffers
                glfwSwapBuffers(window);        
            }      
        }
    }

	for (int j = 0; j < num_piezas; ++j)
	{
		for (int i = 0; i < num_backgrounds; ++i)
		{
			delete models_backgrounds[j][i];
		}
	}
        
    // Terminate GLFW, clearing any resources allocated by GLFW.
    glfwTerminate( );
    
    return 0;
}