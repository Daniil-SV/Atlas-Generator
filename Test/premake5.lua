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
	"../include"
}

links {
	"AtlasGenerator"
}

filter "configurations:Debug"
links {
	"../ThirdParty/lib/opencv/%{cfg.architecture}/%{cfg.system}/static/opencv_world470d"
}
defines { "DEBUG" }
runtime "Debug"
symbols "on"

filter "configurations:Release"
links {
	"../ThirdParty/lib/opencv/%{cfg.architecture}/%{cfg.system}/static/opencv_world470"
}
defines { "NDEBUG" }
runtime "Release"
optimize "on"
