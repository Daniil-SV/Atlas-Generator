project "AtlasGenerator"
kind "StaticLib"

language "C++"
cppdialect "C++17"

objdir "%{wks.location}/build/obj/%{cfg.buildcfg}/%{cfg.system}/%{cfg.architecture}/%{prj.name}"

targetdir "%{wks.location}/build/bin/%{cfg.buildcfg}/%{cfg.system}/%{cfg.architecture}/%{prj.name}"

files {
	"include/**.h",
	"src/**.cpp"
}

includedirs {
	"include",
	"ThirdParty/OpenCV/include",
	"ThirdParty/libnest2d/include"
}

filter "system:windows"
defines { "_WINDOWS" }


filter "configurations:Debug"
links {
	"ThirdParty/libnest2d/lib/%{cfg.architecture}/%{cfg.system}/nloptd",
	"ThirdParty/libnest2d/lib/%{cfg.architecture}/%{cfg.system}/polyclippingd"
}
defines { "DEBUG" }
runtime "Debug"
symbols "on"

filter "configurations:Release"
links {
	"ThirdParty/libnest2d/lib/%{cfg.architecture}/%{cfg.system}/nlopt",
	"ThirdParty/libnest2d/lib/%{cfg.architecture}/%{cfg.system}/polyclipping"
}
defines { "NDEBUG" }
runtime "Release"
optimize "on"
