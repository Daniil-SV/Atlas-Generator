project "AtlasGeneratorTest"
kind "ConsoleApp"

language "C++"
cppdialect "C++17"

objdir "%{wks.location}/build/obj/%{cfg.buildcfg}/%{cfg.system}/%{cfg.architecture}/%{prj.name}"

targetdir "%{wks.location}/build/bin/%{cfg.buildcfg}/%{cfg.system}/%{cfg.architecture}/%{prj.name}"

files {
	"src/**.cpp"
}

includedirs {
	"%{wks.location}/include",
	"%{wks.location}/ThirdParty/OpenCV/include"
}

links {
	"AtlasGenerator"
}

filter "configurations:Debug"
links {
	"%{wks.location}/ThirdParty/OpenCV/lib/%{cfg.architecture}/%{cfg.system}/static/opencv_world470d"
}
defines { "DEBUG" }
runtime "Debug"
symbols "on"

filter "configurations:Release"
links {
	"../ThirdParty/OpenCV/lib/%{cfg.architecture}/%{cfg.system}/static/opencv_world470"
}
defines { "NDEBUG" }
runtime "Release"
optimize "on"
